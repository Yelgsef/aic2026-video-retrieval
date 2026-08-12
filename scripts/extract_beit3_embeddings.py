from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEIT3_SOURCE = PROJECT_ROOT / "third_party" / "unilm" / "beit3"
if str(BEIT3_SOURCE) not in sys.path:
    sys.path.insert(0, str(BEIT3_SOURCE))

import modeling_finetune


ENCODER_ID = "beit3_base_480_vqa_kaggle_v3"
CHECKPOINT = PROJECT_ROOT / "models" / "beit3_vqa_base" / "beit3_base_patch16_480_vqa.pth"


def atomic_json(data: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def find_video(video_id: str) -> Path:
    videos = pd.read_parquet(PROJECT_ROOT / "workspace" / "manifests" / "videos.parquet")
    match = videos.loc[videos["video_id"] == video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one video row for {video_id}, got {len(match)}")
    return Path(match.iloc[0]["video_path"])


def load_model(device: torch.device) -> torch.nn.Module:
    model = modeling_finetune.beit3_base_patch16_480_vqav2()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def encode(images: list[torch.Tensor], model: torch.nn.Module, device: torch.device) -> np.ndarray:
    batch = torch.stack(images).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens = model.beit3(textual_tokens=None, visual_tokens=batch)["encoder_out"]
        features = model.pooler(tokens)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract image embeddings with Kaggle BEiT-3 Base VQA.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sampling", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test only: process first N samples.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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
    model = load_model(device)
    preprocess = transforms.Compose([
        transforms.Resize((480, 480), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    torch.cuda.reset_peak_memory_stats()
    temporary = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(len(samples), 768))
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
                images.append(preprocess(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
                rows.append(row)
                if len(images) >= args.batch_size:
                    flush()
                    if extracted % 256 < args.batch_size:
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
    norms = np.linalg.norm(np.asarray(persisted, dtype=np.float32), axis=1)
    elapsed = time.time() - started
    report = {
        "video_id": args.video_id,
        "encoder_id": ENCODER_ID,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_source": "phong2004/beit-3-vqa-finetune/pyTorch/default/3",
        "pooling": "VQA pooler over image-only encoder output",
        "rows": len(samples),
        "dimension": 768,
        "batch_size": args.batch_size,
        "decoded_frames": decoded,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        "elapsed_seconds": round(elapsed, 3),
        "sample_fps": round(len(samples) / elapsed, 3),
        "outputs": {"embeddings": str(matrix_path), "mapping": str(mapping_path)},
    }
    report_path = PROJECT_ROOT / "workspace" / "reports" / f"beit3_{args.video_id}{suffix}.json"
    atomic_json(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
