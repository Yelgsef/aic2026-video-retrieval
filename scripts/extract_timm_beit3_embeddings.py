from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "hf_hub:timm/beit3_base_patch16_224.pt"
ENCODER_ID = "timm_beit3_base_patch16_224_pt"


def find_video(video_id: str) -> Path:
    videos = pd.read_parquet(PROJECT_ROOT / "workspace" / "manifests" / "videos.parquet")
    match = videos.loc[videos["video_id"] == video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one video row for {video_id}, got {len(match)}")
    return Path(match.iloc[0]["video_path"])


def encode(images: list[torch.Tensor], model: torch.nn.Module, device: torch.device) -> np.ndarray:
    batch = torch.stack(images).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        features = model(batch)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timm BEiT-3 image embeddings.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sampling", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    started = time.time()
    video_path = find_video(args.video_id)
    sampling_path = args.sampling or PROJECT_ROOT / "workspace" / "samples" / f"{args.video_id}.parquet"
    samples = pd.read_parquet(sampling_path).sort_values("frame_id").reset_index(drop=True)
    if args.max_frames is not None:
        samples = samples.iloc[: args.max_frames].copy()
    if samples.empty or samples["frame_id"].duplicated().any():
        raise ValueError("Invalid sampling manifest")

    output_dir = PROJECT_ROOT / "workspace" / "features" / ENCODER_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".smoke{len(samples)}" if args.max_frames is not None else ""
    matrix_path = output_dir / f"{args.video_id}{suffix}.f16.npy"
    mapping_path = output_dir / f"{args.video_id}{suffix}.parquet"
    if matrix_path.exists() and not args.force:
        raise FileExistsError(f"Output exists: {matrix_path}")

    device = torch.device("cuda")
    model = timm.create_model(MODEL_ID, pretrained=True, num_classes=0).to(device).eval()
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    feature_dim = int(model.num_features)
    torch.cuda.reset_peak_memory_stats()

    temporary = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(samples), feature_dim)
    )
    target_to_row = {int(frame): row for row, frame in enumerate(samples["frame_id"])}
    last_target = max(target_to_row)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    images: list[torch.Tensor] = []
    rows: list[int] = []
    extracted = 0
    decoded = 0

    def flush() -> None:
        nonlocal extracted
        if not images:
            return
        matrix[np.asarray(rows, dtype=np.int64)] = encode(images, model, device).astype(np.float16)
        extracted += len(images)
        images.clear()
        rows.clear()

    try:
        frame_id = 0
        while frame_id <= last_target:
            ok, bgr = capture.read()
            if not ok:
                break
            decoded += 1
            row = target_to_row.get(frame_id)
            if row is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(rgb)))
                rows.append(row)
                if len(images) >= args.batch_size:
                    flush()
                    if extracted % 1024 < args.batch_size:
                        print(f"Embedded {extracted:,}/{len(samples):,}", flush=True)
            frame_id += 1
        flush()
    finally:
        capture.release()

    if extracted != len(samples):
        raise RuntimeError(f"Extracted {extracted}/{len(samples)} frames")
    matrix.flush()
    del matrix
    os.replace(temporary, matrix_path)

    mapping = samples.copy()
    mapping.insert(0, "embedding_row", np.arange(len(mapping), dtype=np.int64))
    mapping.insert(1, "encoder_id", ENCODER_ID)
    temporary_mapping = mapping_path.with_suffix(".parquet.tmp")
    mapping.to_parquet(temporary_mapping, index=False)
    os.replace(temporary_mapping, mapping_path)

    persisted = np.load(matrix_path, mmap_mode="r")
    check_idx = np.linspace(0, len(persisted) - 1, min(2000, len(persisted)), dtype=np.int64)
    norms = np.linalg.norm(np.asarray(persisted[check_idx], dtype=np.float32), axis=1)
    elapsed = time.time() - started
    report = {
        "video_id": args.video_id,
        "encoder_id": ENCODER_ID,
        "model_id": MODEL_ID,
        "model_role": "vision-only image feature backbone",
        "rows": len(samples),
        "dimension": feature_dim,
        "batch_size": args.batch_size,
        "decoded_frames": decoded,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        "elapsed_seconds": round(elapsed, 3),
        "sample_fps": round(len(samples) / elapsed, 3),
        "outputs": {"embeddings": str(matrix_path), "mapping": str(mapping_path)},
    }
    report_path = PROJECT_ROOT / "workspace" / "reports" / f"timm_beit3_{args.video_id}{suffix}.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
