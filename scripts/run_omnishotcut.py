from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "omnishotcut" / "OmniShotCut_ckpt.pth"


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
        "--preprocess-device", choices=("cpu", "cuda"), default="cuda",
        help="Run inference preprocessing on CUDA (default) or use the legacy CPU path.",
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

    import omnishotcut

    model = omnishotcut.load(str(checkpoint_path))
    prediction = model.inference(
        str(input_path),
        mode=args.mode,
        overlap=args.overlap,
        decoder_backend=args.decoder_backend,
        prefetch_windows=args.prefetch_windows,
        preprocess_device=args.preprocess_device,
    )
    if args.mode == "clean_shot":
        ranges = prediction
        intra_labels = None
        inter_labels = None
    else:
        ranges, intra_labels, inter_labels = prediction

    decoded_frame_count = model.last_inference_stats.get("decoded_frame_count")
    result = {
        "decoded_frame_count": decoded_frame_count,
        "ranges_end_exclusive": True,
        "ranges": ranges,
        "intra_labels": intra_labels,
        "inter_labels": inter_labels,
    }
    atomic_json(result, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
