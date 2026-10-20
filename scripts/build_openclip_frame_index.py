from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed offline-index frames with an independent OpenCLIP model.")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace/offline_index")
    parser.add_argument("--model", type=Path, default=ROOT / "models/openclip-vit-l14-laion2b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    destination = args.index / "frame_embeddings_openclip.npy"
    keys_destination = args.index / "openclip_frame_keys.parquet"
    report_destination = args.index / "openclip_report.json"
    frames = pd.read_parquet(args.index / "frames.parquet")
    key_columns = ["video_id", "frame_id"]
    required_columns = set(key_columns)
    if not required_columns.issubset(frames.columns):
        raise ValueError(f"frames.parquet missing columns: {sorted(required_columns - set(frames.columns))}")
    if frames.duplicated(key_columns).any():
        raise ValueError("frames.parquet contains duplicate (video_id, frame_id) keys")
    manifest = pd.read_parquet(ROOT / "workspace/manifests/videos.parquet").set_index("video_id")
    if frames.empty:
        raise ValueError("frames.parquet contains no frames")
    source_fingerprints: dict[str, str] = {}
    for video_id in frames.video_id.astype(str).unique():
        if video_id not in manifest.index:
            raise ValueError(f"Video {video_id} is missing from videos.parquet")
        source_path = Path(str(manifest.loc[video_id].video_path)).resolve()
        stat = source_path.stat()
        source_fingerprints[video_id] = hashlib.sha256(
            f"{source_path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = CLIPModel.from_pretrained(str(args.model), dtype=dtype, local_files_only=True).to(device).eval()
    dimension = int(model.config.projection_dim)
    model_files = sorted(path for path in args.model.rglob("*") if path.is_file())
    model_fingerprint = hashlib.sha256("\n".join(
        f"{path.relative_to(args.model)}|{path.stat().st_size}|{path.stat().st_mtime_ns}"
        for path in model_files).encode()).hexdigest()
    reused = 0
    pending = np.ones(len(frames), dtype=bool)
    old_matrix = None
    if destination.exists() and keys_destination.exists() and not args.force:
        old_matrix = np.load(destination, mmap_mode="r")
        old_keys = pd.read_parquet(keys_destination)
        old_report = (json.loads(report_destination.read_text(encoding="utf-8"))
                      if report_destination.exists() else {})
        if (len(old_keys) != len(old_matrix) or old_matrix.shape[1] != dimension
                or old_report.get("model_fingerprint") != model_fingerprint
                or "source_fingerprint" not in old_keys):
            raise ValueError("Existing OpenCLIP matrix and key sidecar are incompatible")
        old_lookup = {
            (str(row.video_id), int(row.frame_id), str(row.source_fingerprint)): index
            for index, row in old_keys.reset_index(drop=True).iterrows()
        }
        for index, row in frames.reset_index(drop=True).iterrows():
            video_id = str(row.video_id)
            old_index = old_lookup.get((video_id, int(row.frame_id), source_fingerprints[video_id]))
            if old_index is not None:
                pending[index] = False
                reused += 1
    elif destination.exists() and not args.force:
        raise FileExistsError(
            f"OpenCLIP index exists without {keys_destination.name}; rerun once with --force "
            "to create an incremental-safe key sidecar"
        )
    started, decoded = time.perf_counter(), 0
    args.index.mkdir(parents=True, exist_ok=True)
    staging_context = tempfile.TemporaryDirectory(prefix=".openclip_staging_", dir=args.index)
    staging = Path(staging_context.name)
    staged_matrix_path = staging / destination.name
    matrix = np.lib.format.open_memmap(
        staged_matrix_path, mode="w+", dtype=np.float32, shape=(len(frames), dimension))
    if old_matrix is not None:
        for index, row in frames.reset_index(drop=True).iterrows():
            if not pending[index]:
                video_id = str(row.video_id)
                old_index = old_lookup[(video_id, int(row.frame_id), source_fingerprints[video_id])]
                matrix[index] = old_matrix[old_index]

    def encode(images: list[Image.Image]) -> np.ndarray:
        try:
            inputs = processor(images=images, return_tensors="pt").to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
            with torch.inference_mode():
                values = model.get_image_features(**inputs).float()
                values = torch.nn.functional.normalize(values, dim=-1)
            return values.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(images) == 1:
                raise
            middle = len(images) // 2
            return np.concatenate((encode(images[:middle]), encode(images[middle:])), axis=0)

    for video_id, group in frames.groupby("video_id", sort=False):
        capture = cv2.VideoCapture(str(manifest.loc[video_id].video_path))
        batch_images: list[Image.Image] = []
        batch_rows: list[int] = []
        group_decoded = int(pending[group.index].sum())

        def flush() -> None:
            nonlocal decoded
            if not batch_images:
                return
            matrix[batch_rows] = encode(batch_images)
            decoded += len(batch_images)
            batch_images.clear()
            batch_rows.clear()

        for row_index, row in group.sort_values("frame_id").iterrows():
            if not pending[int(row_index)]:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame_id))
            ok, image = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f"Could not decode {video_id} frame {row.frame_id}")
            batch_images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            batch_rows.append(int(row_index))
            if len(batch_images) >= args.batch_size:
                flush()
        flush()
        capture.release()
        print(json.dumps({"video_id": video_id, "decoded": group_decoded}), flush=True)
    matrix.flush()
    del matrix
    if old_matrix is not None:
        del old_matrix
    key_table = frames[key_columns].reset_index(drop=True).copy()
    key_table["source_fingerprint"] = key_table.video_id.astype(str).map(source_fingerprints)
    staged_keys_path = staging / keys_destination.name
    key_table.to_parquet(staged_keys_path, index=False)
    report = {"schema_version": "0.2.0", "model": str(args.model),
              "model_fingerprint": model_fingerprint, "source_fingerprints": source_fingerprints,
              "frames": len(frames), "decoded_frames": decoded, "reused_frames": reused,
              "dimension": dimension, "device": str(device),
              "elapsed_seconds": time.perf_counter() - started, "output": str(destination)}
    staged_report_path = staging / report_destination.name
    staged_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_destination.unlink(missing_ok=True)
    staged_matrix_path.replace(destination)
    staged_keys_path.replace(keys_destination)
    staged_report_path.replace(report_destination)
    staging_context.cleanup()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
