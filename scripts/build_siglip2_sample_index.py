from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.siglip_retrieval import encoder_id_from_model, save_index
from aic_pipeline.keyframe_selection import select_video_keyframes
from aic_pipeline.paths import PipelinePaths


def find_video(video_id: str) -> Path:
    videos = pd.read_parquet(PROJECT_ROOT / "workspace" / "manifests" / "videos.parquet")
    rows = videos[videos["video_id"] == video_id]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one video for {video_id}, got {len(rows)}")
    return Path(rows.iloc[0]["video_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed every sampled frame directly from video with SigLIP 2.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--unified-select", action="store_true",
        help="Reuse the sampled-frame SigLIP vectors for clustering and the compact retrieval index.",
    )
    parser.add_argument(
        "--reuse-dense-features", "--reuse-dense-index",
        dest="reuse_dense_features", action="store_true",
        help="Skip inference and build unified artifacts from existing dense features.",
    )
    parser.add_argument("--k-cap", type=int, default=12)
    parser.add_argument("--redundancy-cosine", type=float, default=0.95)
    parser.add_argument("--temporal-window-frames", type=int, default=15)
    parser.add_argument("--minimum-frames-per-shot", type=int, default=3)
    parser.add_argument("--protected-anchor-policy", choices=("none", "core", "all"), default="none")
    parser.add_argument("--adjacent-shot-cosine", type=float, default=0.995)
    parser.add_argument("--global-within-shot-cosine", type=float, default=0.995)
    parser.add_argument("--single-cluster-max-distance", type=float, default=0.01)
    parser.add_argument("--temporal-chunk-frames", type=int, default=67)
    parser.add_argument("--temporal-chunk-overlap", type=int, default=10)
    parser.add_argument("--max-medoids-per-chunk", type=int, choices=(1, 2), default=2)
    parser.add_argument("--coverage-window-frames", type=int, default=103)
    parser.add_argument("--evidence-manifest", type=Path)
    args = parser.parse_args()

    samples_path = PROJECT_ROOT / "workspace" / "samples" / f"{args.video_id}.parquet"
    samples = pd.read_parquet(samples_path).sort_values("frame_id").reset_index(drop=True)
    if samples.empty or samples["frame_id"].duplicated().any():
        raise ValueError("Sample manifest is empty or has duplicate frame IDs")
    video_path = find_video(args.video_id)
    encoder_id = encoder_id_from_model(args.model_id)
    feature_dir = PROJECT_ROOT / "workspace" / "features" / encoder_id
    feature_matrix_path = feature_dir / f"{args.video_id}.f16.npy"
    feature_mapping_path = feature_dir / f"{args.video_id}.parquet"
    if feature_matrix_path.exists() and not (args.force or args.reuse_dense_features):
        raise FileExistsError(f"Dense features exist: {feature_matrix_path}")

    if args.reuse_dense_features:
        if not args.unified_select:
            raise ValueError("--reuse-dense-features requires --unified-select")
        requested_frame_ids = samples.frame_id.astype(np.int64).to_numpy()
        embeddings = np.load(feature_matrix_path, mmap_mode="r")
        dense_metadata = pd.read_parquet(feature_mapping_path).sort_values("embedding_row").reset_index(drop=True)
        if len(dense_metadata) != len(embeddings):
            raise ValueError("Dense feature metadata does not match embeddings")
        if not np.array_equal(requested_frame_ids, dense_metadata.frame_id.astype(np.int64).to_numpy()):
            raise ValueError("Dense features were built from a different sample manifest")
        samples["embedding_row"] = dense_metadata["embedding_row"].to_numpy(dtype=np.int64)
        samples["encoder_id"] = encoder_id
        config = {
            "model_id": args.model_id,
            "encoder_id": encoder_id,
            "index_kind": "stride_samples",
            "video_id": args.video_id,
            "count": len(samples),
            "dimension": int(embeddings.shape[1]),
            "normalized": True,
            "similarity": "cosine_via_inner_product",
            "source_manifest": str(samples_path),
        }
        print(f"Reusing {len(embeddings):,} existing dense feature vectors", flush=True)
    else:
        config = None

    if not args.reuse_dense_features:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=True, use_fast=False)
        model = AutoModel.from_pretrained(args.model_id, dtype=dtype, local_files_only=True).to(device).eval()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        target_rows = list(samples.itertuples(index=False))
        target_position = 0
        frame_id = 0
        batch_images: list[Image.Image] = []
        chunks: list[np.ndarray] = []
        decoded = 0
        def flush() -> None:
            if not batch_images:
                return
            def encode(images: list[Image.Image]) -> torch.Tensor:
                try:
                    inputs = processor(images=images, return_tensors="pt").to(device)
                    with torch.inference_mode():
                        result = model.get_image_features(**inputs)
                        return torch.nn.functional.normalize(result.float(), dim=-1).cpu()
                except torch.cuda.OutOfMemoryError:
                    if len(images) == 1:
                        raise
                    torch.cuda.empty_cache()
                    midpoint = len(images) // 2
                    print(
                        f"CUDA OOM at batch {len(images)}; retrying as {midpoint}+{len(images) - midpoint}",
                        flush=True,
                    )
                    return torch.cat((encode(images[:midpoint]), encode(images[midpoint:])), dim=0)

            chunks.append(encode(batch_images).numpy())
            batch_images.clear()

        try:
            while target_position < len(target_rows):
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                target_frame = int(target_rows[target_position].frame_id)
                if frame_id == target_frame:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_images.append(Image.fromarray(rgb))
                    target_position += 1
                    if len(batch_images) >= args.batch_size:
                        flush()
                    if target_position % 250 == 0 or target_position == len(target_rows):
                        print(f"Embedded {target_position:,}/{len(target_rows):,}", flush=True)
                frame_id += 1
            flush()
        finally:
            capture.release()
        if target_position != len(target_rows):
            raise RuntimeError(f"Decoded only {target_position}/{len(target_rows)} target frames")

        embeddings = np.concatenate(chunks)
        config = {
            "model_id": args.model_id,
            "encoder_id": encoder_id,
            "index_kind": "stride_samples",
            "video_id": args.video_id,
            "count": len(samples),
            "dimension": int(embeddings.shape[1]),
            "normalized": True,
            "similarity": "cosine_via_inner_product",
            "source_manifest": str(samples_path),
        }

        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_mapping = samples.copy().reset_index(drop=True)
        feature_mapping["embedding_row"] = np.arange(len(feature_mapping), dtype=np.int64)
        feature_mapping["encoder_id"] = encoder_id
        feature_mapping.to_parquet(feature_mapping_path, index=False)
        np.save(feature_matrix_path, np.asarray(embeddings, dtype=np.float16))
        del embeddings

    if args.unified_select:
        selection = select_video_keyframes(
            PipelinePaths.from_root(PROJECT_ROOT),
            video_id=args.video_id,
            encoder_id=encoder_id,
            k_cap=args.k_cap,
            redundancy_cosine=args.redundancy_cosine,
            temporal_window_frames=args.temporal_window_frames,
            minimum_frames_per_shot=args.minimum_frames_per_shot,
            protected_anchor_policy=args.protected_anchor_policy,
            adjacent_shot_cosine=args.adjacent_shot_cosine,
            global_within_shot_cosine=args.global_within_shot_cosine,
            single_cluster_max_distance=args.single_cluster_max_distance,
            temporal_chunk_frames=args.temporal_chunk_frames,
            temporal_chunk_overlap=args.temporal_chunk_overlap,
            max_medoids_per_chunk=args.max_medoids_per_chunk,
            coverage_window_frames=args.coverage_window_frames,
            evidence_manifest_path=args.evidence_manifest,
            force=args.force,
        )
        selected_path = PROJECT_ROOT / "workspace" / "keyframes" / encoder_id / f"{args.video_id}.parquet"
        selected = pd.read_parquet(selected_path)
        dense_vectors = np.load(feature_matrix_path, mmap_mode="r")
        compact = dense_vectors[selected["embedding_row"].to_numpy(dtype=np.int64)]
        compact_dir = PROJECT_ROOT / "workspace" / "retrieval" / encoder_id / f"{args.video_id}_unified"
        compact_config = {
            **config,
            "index_kind": "adaptive_keyframes",
            "count": len(selected),
            "source_manifest": str(selected_path),
            "selection": selection["parameters"],
            "single_visual_encoder_pass": True,
        }
        save_index(compact_dir, compact, selected, compact_config)
        config = {**config, "compact_index": str(compact_dir)}

    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
