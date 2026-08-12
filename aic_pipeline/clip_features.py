from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import atomic_write_json, atomic_write_parquet
from .paths import PipelinePaths


def pack_provided_clip(
    paths: PipelinePaths,
    output_dtype: str = "float16",
    progress_every: int = 100,
) -> dict[str, object]:
    """Normalize and concatenate per-video provided CLIP features.

    Rows are ordered by video_id and local clip_row. The companion Parquet
    mapping is the only supported way to translate global rows back to frames.
    """
    started = time.time()
    paths.ensure_output_dirs()

    manifest_path = paths.manifests / "provided_keyframes.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run scripts/audit_dataset.py first."
        )

    frames = pd.read_parquet(manifest_path)
    frames = frames.loc[frames["clip_row"].notna()].copy()
    frames["clip_row"] = frames["clip_row"].astype("int64")
    frames = frames.sort_values(["video_id", "clip_row"]).reset_index(drop=True)

    video_ids = frames["video_id"].drop_duplicates().tolist()
    dimensions: set[int] = set()
    row_counts: dict[str, int] = {}
    for video_id in video_ids:
        feature_path = paths.archive / "clip-features-32" / f"{video_id}.npy"
        features = np.load(feature_path, mmap_mode="r")
        if features.ndim != 2:
            raise ValueError(f"Invalid feature shape for {video_id}: {features.shape}")
        row_counts[video_id] = int(features.shape[0])
        dimensions.add(int(features.shape[1]))
        del features

    if len(dimensions) != 1:
        raise ValueError(f"Provided CLIP dimensions are inconsistent: {dimensions}")
    dimension = dimensions.pop()
    expected_rows = int(sum(row_counts.values()))
    if expected_rows != len(frames):
        raise ValueError(
            f"Manifest rows ({len(frames)}) != feature rows ({expected_rows})"
        )

    output_dir = paths.features / "provided_clip_vit_b32"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "embeddings.f16.npy"
    temporary_matrix = output_dir / "embeddings.f16.npy.tmp"
    mapping_path = output_dir / "row_mapping.parquet"

    target_dtype = np.dtype(output_dtype)
    matrix = np.lib.format.open_memmap(
        temporary_matrix,
        mode="w+",
        dtype=target_dtype,
        shape=(expected_rows, dimension),
    )

    global_offset = 0
    mapping_parts: list[pd.DataFrame] = []
    zero_norm_rows = 0
    for index, video_id in enumerate(video_ids, start=1):
        feature_path = paths.archive / "clip-features-32" / f"{video_id}.npy"
        source = np.asarray(np.load(feature_path, mmap_mode="r"), dtype=np.float32)
        norms = np.linalg.norm(source, axis=1, keepdims=True)
        zero_norm_rows += int(np.count_nonzero(norms.squeeze(1) <= 1e-12))
        normalized = source / np.clip(norms, 1e-12, None)
        end = global_offset + len(normalized)
        matrix[global_offset:end] = normalized.astype(target_dtype, copy=False)

        local_mapping = frames.loc[frames["video_id"] == video_id].copy()
        if len(local_mapping) != len(normalized):
            raise ValueError(
                f"{video_id}: mapping rows {len(local_mapping)} != features {len(normalized)}"
            )
        local_mapping.insert(0, "global_row", np.arange(global_offset, end))
        mapping_parts.append(local_mapping)
        global_offset = end

        if progress_every and (index % progress_every == 0 or index == len(video_ids)):
            print(f"Packed {index}/{len(video_ids)} videos ({global_offset:,} rows)")

    matrix.flush()
    del matrix
    os.replace(temporary_matrix, matrix_path)

    mapping = pd.concat(mapping_parts, ignore_index=True)
    atomic_write_parquet(mapping, mapping_path)

    # Reload a small sample to verify the on-disk matrix and normalization.
    persisted = np.load(matrix_path, mmap_mode="r")
    sample_size = min(10_000, len(persisted))
    sample_indices = np.linspace(0, len(persisted) - 1, sample_size, dtype=np.int64)
    sample = np.asarray(persisted[sample_indices], dtype=np.float32)
    sample_norms = np.linalg.norm(sample, axis=1)

    report = {
        "pipeline_version": "0.1.0",
        "encoder_id": "provided_clip_vit_b32",
        "source_videos": len(video_ids),
        "rows": expected_rows,
        "dimension": dimension,
        "dtype": str(target_dtype),
        "zero_norm_rows": zero_norm_rows,
        "sample_norm_min": float(sample_norms.min()),
        "sample_norm_max": float(sample_norms.max()),
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": {
            "embeddings": str(matrix_path),
            "row_mapping": str(mapping_path),
        },
    }
    atomic_write_json(report, paths.reports / "provided_clip_pack.json")
    return report
