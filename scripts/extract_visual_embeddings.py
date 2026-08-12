from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODER_ID = "openclip_vit_b32_openai_quickgelu"


def atomic_json(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def find_video(video_id: str) -> Path:
    videos_manifest = pd.read_parquet(PROJECT_ROOT / "workspace" / "manifests" / "videos.parquet")
    match = videos_manifest.loc[videos_manifest["video_id"] == video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one video manifest row for {video_id}, got {len(match)}")
    return Path(match.iloc[0]["video_path"])


def encode_batch(
    images: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.stack(images).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream sampled video frames through OpenCLIP ViT-B/32."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sampling", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for visual embedding extraction")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    video_id = args.video_id
    video_path = find_video(video_id)
    sampling_path = args.sampling or (
        PROJECT_ROOT / "workspace" / "samples" / f"{video_id}.parquet"
    )
    samples = pd.read_parquet(sampling_path).sort_values("frame_id").reset_index(drop=True)
    if samples.empty:
        raise ValueError(f"Sampling manifest is empty: {sampling_path}")
    if samples["frame_id"].duplicated().any():
        raise ValueError("Sampling manifest contains duplicate frame IDs")

    output_dir = PROJECT_ROOT / "workspace" / "features" / ENCODER_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"{video_id}.f16.npy"
    mapping_path = output_dir / f"{video_id}.parquet"
    report_path = PROJECT_ROOT / "workspace" / "reports" / f"visual_{video_id}.json"
    if matrix_path.exists() and mapping_path.exists() and not args.force:
        print(f"Skip existing embeddings: {matrix_path}")
        return

    started = time.time()
    device = torch.device("cuda")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu",
        pretrained="openai",
        device=device,
    )
    model.eval()
    feature_dim = int(model.visual.output_dim)
    torch.cuda.reset_peak_memory_stats()

    temporary_matrix = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    matrix = np.lib.format.open_memmap(
        temporary_matrix,
        mode="w+",
        dtype=np.float16,
        shape=(len(samples), feature_dim),
    )

    target_to_row = {
        int(frame_id): int(row_index)
        for row_index, frame_id in enumerate(samples["frame_id"].tolist())
    }
    last_target = max(target_to_row)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    batch_images: list[torch.Tensor] = []
    batch_rows: list[int] = []
    decoded_frames = 0
    extracted_frames = 0

    def flush() -> None:
        nonlocal extracted_frames
        if not batch_images:
            return
        encoded = encode_batch(batch_images, model, device)
        matrix[np.asarray(batch_rows, dtype=np.int64)] = encoded.astype(np.float16)
        extracted_frames += len(batch_images)
        batch_images.clear()
        batch_rows.clear()

    try:
        frame_id = 0
        while frame_id <= last_target:
            ok, bgr = capture.read()
            if not ok:
                break
            decoded_frames += 1
            output_row = target_to_row.get(frame_id)
            if output_row is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                batch_images.append(preprocess(Image.fromarray(rgb)))
                batch_rows.append(output_row)
                if len(batch_images) >= args.batch_size:
                    flush()
                    if extracted_frames % 1024 < args.batch_size:
                        print(f"Embedded {extracted_frames:,}/{len(samples):,} frames")
            frame_id += 1
        flush()
    finally:
        capture.release()

    if extracted_frames != len(samples):
        del matrix
        raise RuntimeError(
            f"Extracted {extracted_frames}/{len(samples)} sampled frames; "
            f"decoded through frame {decoded_frames - 1}"
        )

    matrix.flush()
    del matrix
    os.replace(temporary_matrix, matrix_path)

    mapping = samples.copy()
    mapping.insert(0, "embedding_row", np.arange(len(mapping), dtype=np.int64))
    mapping.insert(1, "encoder_id", ENCODER_ID)
    mapping.to_parquet(mapping_path.with_suffix(".parquet.tmp"), index=False, engine="pyarrow")
    os.replace(mapping_path.with_suffix(".parquet.tmp"), mapping_path)

    persisted = np.load(matrix_path, mmap_mode="r")
    check_indices = np.linspace(0, len(persisted) - 1, min(2000, len(persisted)), dtype=np.int64)
    check = np.asarray(persisted[check_indices], dtype=np.float32)
    norms = np.linalg.norm(check, axis=1)
    elapsed = time.time() - started
    report = {
        "pipeline_version": "0.1.0",
        "video_id": video_id,
        "video_path": str(video_path),
        "sampling_manifest": str(Path(sampling_path).resolve()),
        "encoder_id": ENCODER_ID,
        "model": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "rows": len(samples),
        "dimension": feature_dim,
        "dtype": "float16",
        "batch_size": args.batch_size,
        "decoded_frames": decoded_frames,
        "sample_norm_min": float(norms.min()),
        "sample_norm_max": float(norms.max()),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        "elapsed_seconds": round(elapsed, 3),
        "frames_per_second": round(extracted_frames / elapsed, 3),
        "outputs": {
            "embeddings": str(matrix_path),
            "mapping": str(mapping_path),
        },
    }
    atomic_json(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
