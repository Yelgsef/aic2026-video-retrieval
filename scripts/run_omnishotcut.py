from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "omnishotcut" / "OmniShotCut_ckpt.pth"
LOCAL_FFMPEG_DIR = PROJECT_ROOT / "tools" / "ffmpeg"


def configure_ffmpeg() -> None:
    ffmpeg_executable = LOCAL_FFMPEG_DIR / "ffmpeg.exe"
    if ffmpeg_executable.exists():
        os.environ["PATH"] = f"{LOCAL_FFMPEG_DIR}{os.pathsep}{os.environ.get('PATH', '')}"


def read_prefix_frames(video_path: Path, frame_limit: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        for _ in range(frame_limit):
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Decoded zero frames from: {video_path}")
    return np.stack(frames), fps


def atomic_json(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniShotCut on one AIC video.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["default", "clean_shot"], default="default")
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument(
        "--decoder-backend", choices=("auto", "cpu", "nvdec"), default="auto",
        help="auto uses NVDEC when available and otherwise falls back to the legacy CPU decoder.",
    )
    parser.add_argument(
        "--prefetch-windows", type=int, default=3,
        help="Decoded model windows buffered while CUDA inference runs.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Only decode the first N frames. Intended for smoke tests.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output or (
        PROJECT_ROOT / "workspace" / "shots" / f"{input_path.stem}.json"
    )
    output_path = output_path.resolve()
    if output_path.exists() and not args.force:
        print(f"Skip existing output: {output_path}")
        return
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if args.prefetch_windows <= 0:
        raise ValueError("--prefetch-windows must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("OmniShotCut currently requires a CUDA-enabled PyTorch runtime.")

    configure_ffmpeg()
    import omnishotcut

    started = time.time()
    fps = None
    frame_count = None
    if args.frame_limit:
        model_input, fps = read_prefix_frames(input_path, args.frame_limit)
        frame_count = len(model_input)
    else:
        model_input = str(input_path)

    model = omnishotcut.load(str(checkpoint_path))
    torch.cuda.reset_peak_memory_stats()
    inference_started = time.time()
    prediction = model.inference(
        model_input,
        mode=args.mode,
        overlap=args.overlap,
        decoder_backend=args.decoder_backend,
        prefetch_windows=args.prefetch_windows,
    )
    inference_seconds = time.time() - inference_started

    if args.mode == "clean_shot":
        ranges = prediction
        intra_labels = None
        inter_labels = None
    else:
        ranges, intra_labels, inter_labels = prediction

    result = {
        "pipeline_version": "0.1.0",
        "detector": "omnishotcut",
        "input_video": str(input_path),
        "checkpoint": str(checkpoint_path),
        "mode": args.mode,
        "overlap": args.overlap,
        "decoder_backend_requested": args.decoder_backend,
        "decoder": model.last_inference_stats,
        "frame_limit": args.frame_limit,
        "decoded_frame_count": (
            frame_count if frame_count is not None
            else model.last_inference_stats.get("decoded_frame_count")
        ),
        "fps": fps,
        "ranges_end_exclusive": True,
        "ranges": ranges,
        "intra_labels": intra_labels,
        "inter_labels": inter_labels,
        "shot_count": len(ranges),
        "inference_seconds": round(inference_seconds, 3),
        "total_seconds": round(time.time() - started, 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
    }
    atomic_json(result, output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
