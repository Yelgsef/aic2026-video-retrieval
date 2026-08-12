from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import atomic_write_json, atomic_write_parquet
from .paths import PipelinePaths


def _spherical_kmeans(
    vectors: np.ndarray,
    k: int,
    seed: int,
    iterations: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic spherical k-means implementation."""
    n = len(vectors)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    chosen = [first]
    nearest_distance = 1.0 - vectors @ vectors[first]
    for _ in range(1, k):
        weights = np.maximum(nearest_distance, 0.0) ** 2
        if float(weights.sum()) <= 1e-12:
            candidate = next(i for i in range(n) if i not in chosen)
        else:
            candidate = int(rng.choice(n, p=weights / weights.sum()))
        chosen.append(candidate)
        nearest_distance = np.minimum(
            nearest_distance, 1.0 - vectors @ vectors[candidate]
        )

    centers = vectors[chosen].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iterations):
        new_labels = np.argmax(vectors @ centers.T, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = vectors[new_labels == cluster_id]
            if len(members):
                center = members.mean(axis=0)
                norm = np.linalg.norm(center)
                if norm > 1e-12:
                    new_centers[cluster_id] = center / norm
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def _sampled_silhouette(
    vectors: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 256,
) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(vectors):
        return -1.0
    indices = np.linspace(
        0, len(vectors) - 1, min(max_samples, len(vectors)), dtype=np.int64
    )
    distances = 1.0 - vectors[indices] @ vectors.T
    scores: list[float] = []
    for row, source_index in enumerate(indices):
        own = labels[source_index]
        own_mask = labels == own
        own_count = int(own_mask.sum())
        a = (
            float(distances[row, own_mask].sum()) / (own_count - 1)
            if own_count > 1
            else 0.0
        )
        b = min(float(distances[row, labels == other].mean()) for other in unique if other != own)
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 1e-12 else 0.0)
    return float(np.mean(scores))


def _cluster_shot(
    vectors: np.ndarray,
    seed: int,
    k_cap: int,
    single_cluster_max_distance: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    n = len(vectors)
    one_center = vectors.mean(axis=0, keepdims=True)
    one_center /= np.clip(np.linalg.norm(one_center, axis=1, keepdims=True), 1e-12, None)
    one_cluster_distance = float(np.mean(1.0 - vectors @ one_center[0]))
    if n < 3 or one_cluster_distance <= single_cluster_max_distance:
        labels = np.zeros(n, dtype=np.int32)
        return labels, one_center, 1, 0.0

    k_max = min(int(math.ceil(math.sqrt(n))), k_cap, n - 1)
    best: tuple[float, np.ndarray, np.ndarray, int, float] | None = None
    for k in range(2, k_max + 1):
        labels, centers = _spherical_kmeans(vectors, k, seed + k)
        if len(np.unique(labels)) != k:
            continue
        silhouette = _sampled_silhouette(vectors, labels)
        # A small complexity penalty prevents fragmenting smooth shots into many clusters.
        objective = silhouette - 0.015 * (k - 2)
        if best is None or objective > best[0]:
            best = (objective, labels, centers, k, silhouette)
    if best is None:
        labels = np.zeros(n, dtype=np.int32)
        return labels, one_center, 1, 0.0
    return best[1], best[2], best[3], best[4]


def select_video_keyframes(
    paths: PipelinePaths,
    video_id: str,
    encoder_id: str = "openclip_vit_b32_openai_quickgelu",
    k_cap: int = 12,
    redundancy_cosine: float = 0.985,
    temporal_window_frames: int = 15,
    minimum_frames_per_shot: int = 3,
    protected_anchor_policy: str = "core",
    adjacent_shot_cosine: float | None = None,
    global_within_shot_cosine: float | None = None,
    single_cluster_max_distance: float = 0.08,
    force: bool = False,
    report_output_path: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    feature_dir = paths.features / encoder_id
    matrix_path = feature_dir / f"{video_id}.f16.npy"
    mapping_path = feature_dir / f"{video_id}.parquet"
    output_dir = paths.workspace / "keyframes" / encoder_id
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / f"{video_id}.parquet"
    audit_path = output_dir / f"{video_id}.audit.parquet"
    compact_path = output_dir / f"{video_id}.f16.npy"
    report_path = report_output_path or paths.reports / f"keyframes_{video_id}.json"
    if selected_path.exists() and not force:
        raise FileExistsError(f"Output exists: {selected_path}. Use --force to rebuild.")

    mapping = pd.read_parquet(mapping_path).sort_values("embedding_row").reset_index(drop=True)
    embeddings = np.load(matrix_path, mmap_mode="r")
    if len(mapping) != len(embeddings):
        raise ValueError("Embedding matrix and mapping row counts differ")
    vectors = np.asarray(embeddings, dtype=np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    if protected_anchor_policy not in {"none", "core", "all"}:
        raise ValueError("protected_anchor_policy must be 'none', 'core', or 'all'")

    core_anchor_names = {"shot_start", "shot_middle", "shot_end"}
    coverage_anchors = mapping["sampling_reason"].map(
        lambda value: any(reason.startswith("coverage_") for reason in str(value).split("|"))
    )
    if protected_anchor_policy == "none":
        mapping["is_protected_anchor"] = coverage_anchors
    elif protected_anchor_policy == "all":
        mapping["is_protected_anchor"] = mapping["is_mandatory"].astype(bool) | coverage_anchors
    else:
        mapping["is_protected_anchor"] = mapping["sampling_reason"].map(
            lambda value: bool(core_anchor_names.intersection(str(value).split("|")))
        ) | coverage_anchors
    mapping["is_coverage_anchor"] = coverage_anchors

    audit_parts: list[pd.DataFrame] = []
    selected_rows: set[int] = set()
    medoid_rows: set[int] = set()
    shot_stats: list[dict[str, Any]] = []
    for shot_id, shot in mapping.groupby("shot_id", sort=True):
        positions = shot.index.to_numpy(dtype=np.int64)
        shot_vectors = vectors[positions]
        labels, centers, chosen_k, silhouette = _cluster_shot(
            shot_vectors,
            seed=int(shot_id) * 1009 + 17,
            k_cap=k_cap,
            single_cluster_max_distance=single_cluster_max_distance,
        )
        local_medoids: set[int] = set()
        for cluster_id in range(chosen_k):
            members = np.flatnonzero(labels == cluster_id)
            if not len(members):
                continue
            medoid_local = int(members[np.argmax(shot_vectors[members] @ centers[cluster_id])])
            row = int(positions[medoid_local])
            local_medoids.add(row)
            medoid_rows.add(row)
        protected = set(
            positions[shot["is_protected_anchor"].to_numpy(dtype=bool)].tolist()
        )
        candidates = protected | local_medoids
        selected_rows.update(candidates)
        part = shot.copy()
        part["cluster_id"] = labels
        part["shot_k"] = chosen_k
        part["shot_silhouette"] = silhouette
        part["is_cluster_medoid"] = part.index.isin(local_medoids)
        audit_parts.append(part)
        shot_stats.append(
            {"shot_id": int(shot_id), "samples": len(shot), "k": chosen_k,
             "silhouette": silhouette, "candidates": len(candidates)}
        )

    audit = pd.concat(audit_parts).sort_index()
    audit["selection_stage"] = np.where(
        audit.index.isin(medoid_rows), "cluster_medoid", "not_cluster_medoid"
    )
    audit.loc[audit["is_protected_anchor"], "selection_stage"] = "protected_anchor"

    # Conservative redundancy pass, shot-local and chronological.
    removed_to: dict[int, int] = {}
    adjacent_removed_to: dict[int, int] = {}
    global_within_removed_to: dict[int, int] = {}
    final_rows: set[int] = set()
    for shot_id, shot in audit[audit.index.isin(selected_rows)].groupby("shot_id", sort=True):
        ordered = shot.sort_values("frame_id")
        kept: list[int] = []
        for row in ordered.index.to_list():
            row = int(row)
            if bool(audit.at[row, "is_protected_anchor"]):
                kept.append(row)
                continue
            duplicate_of = None
            for previous in reversed(kept):
                delta = int(audit.at[row, "frame_id"] - audit.at[previous, "frame_id"])
                if delta > temporal_window_frames:
                    break
                cosine = float(vectors[row] @ vectors[previous])
                if cosine >= redundancy_cosine:
                    duplicate_of = previous
                    break
            if duplicate_of is None:
                kept.append(row)
            else:
                removed_to[row] = duplicate_of

        target_minimum = min(minimum_frames_per_shot, len(ordered))
        if len(kept) < target_minimum:
            removed = [int(row) for row in ordered.index if int(row) not in kept]
            removed.sort(
                key=lambda row: max(float(vectors[row] @ vectors[x]) for x in kept)
                if kept else -1.0
            )
            for row in removed[: target_minimum - len(kept)]:
                kept.append(row)
                removed_to.pop(row, None)
        final_rows.update(kept)

    # Compare all retained medoids inside each shot. This catches long-range
    # duplicates that the temporal-window pass intentionally does not inspect.
    if global_within_shot_cosine is not None:
        for shot_id in sorted(int(value) for value in audit["shot_id"].unique()):
            current_rows = sorted(
                (row for row in final_rows if int(audit.at[row, "shot_id"]) == shot_id),
                key=lambda row: int(audit.at[row, "frame_id"]),
            )
            kept_rows: list[int] = []
            for row in current_rows:
                if not kept_rows or bool(audit.at[row, "is_protected_anchor"]):
                    kept_rows.append(row)
                    continue
                similarities = vectors[kept_rows] @ vectors[row]
                best_offset = int(np.argmax(similarities))
                if float(similarities[best_offset]) >= global_within_shot_cosine:
                    representative = int(kept_rows[best_offset])
                    final_rows.remove(row)
                    removed_to[row] = representative
                    global_within_removed_to[row] = representative
                else:
                    kept_rows.append(row)

    # Optional compact-index deduplication across immediately adjacent shots.
    # Keep at least one representative per shot and never remove protected anchors.
    if adjacent_shot_cosine is not None:
        shot_ids = sorted(int(value) for value in audit["shot_id"].unique())
        for previous_shot_id, shot_id in zip(shot_ids, shot_ids[1:]):
            if shot_id != previous_shot_id + 1:
                continue
            previous_rows = sorted(
                (row for row in final_rows if int(audit.at[row, "shot_id"]) == previous_shot_id),
                key=lambda row: int(audit.at[row, "frame_id"]),
            )
            current_rows = sorted(
                (row for row in final_rows if int(audit.at[row, "shot_id"]) == shot_id),
                key=lambda row: int(audit.at[row, "frame_id"]),
            )
            if not previous_rows or len(current_rows) <= 1:
                continue
            for row in list(current_rows):
                if len(current_rows) <= 1:
                    break
                if bool(audit.at[row, "is_protected_anchor"]):
                    continue
                similarities = vectors[previous_rows] @ vectors[row]
                best_offset = int(np.argmax(similarities))
                best_similarity = float(similarities[best_offset])
                if best_similarity >= adjacent_shot_cosine:
                    representative = int(previous_rows[best_offset])
                    final_rows.remove(row)
                    current_rows.remove(row)
                    removed_to[row] = representative
                    adjacent_removed_to[row] = representative

    audit["selected"] = audit.index.isin(final_rows)
    audit["redundant_to_embedding_row"] = pd.Series(removed_to, dtype="Int64")
    audit.loc[audit.index.isin(removed_to), "selection_stage"] = "redundancy_removed"
    audit.loc[audit.index.isin(adjacent_removed_to), "selection_stage"] = "adjacent_shot_removed"
    audit.loc[audit.index.isin(global_within_removed_to), "selection_stage"] = "global_within_shot_removed"
    audit = audit.reset_index(drop=True)

    selected = audit[audit["selected"]].sort_values("frame_id").copy()
    expected_coverage_rows = set(
        mapping.index[mapping["is_coverage_anchor"]].astype(int).tolist()
    )
    missing_coverage_rows = expected_coverage_rows - final_rows
    if missing_coverage_rows:
        missing_frames = mapping.loc[sorted(missing_coverage_rows), "frame_id"].astype(int).tolist()
        raise RuntimeError(
            "Keyframe selection removed hard coverage anchors at frames: "
            f"{missing_frames[:20]}"
        )
    selected.insert(0, "keyframe_row", np.arange(len(selected), dtype=np.int64))
    selected["selection_reasons"] = selected.apply(
        lambda row: "|".join(
            reason for reason, enabled in (
                ("protected_anchor", bool(row["is_protected_anchor"])),
                ("coverage_anchor", bool(row["is_coverage_anchor"])),
                ("cluster_medoid", bool(row["is_cluster_medoid"])),
            ) if enabled
        ), axis=1
    )
    compact = vectors[selected["embedding_row"].to_numpy(dtype=np.int64)].astype(np.float16)

    atomic_write_parquet(selected, selected_path)
    atomic_write_parquet(audit, audit_path)
    temporary = compact_path.with_suffix(compact_path.suffix + ".tmp")
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=compact.shape)
    matrix[:] = compact
    matrix.flush()
    del matrix
    temporary.replace(compact_path)

    shot_frame_counts = selected.groupby("shot_id").size()
    report = {
        "pipeline_version": "0.1.0",
        "video_id": video_id,
        "encoder_id": encoder_id,
        "input_samples": len(mapping),
        "shot_count": int(mapping["shot_id"].nunique()),
        "cluster_medoids": len(medoid_rows),
        "sampled_mandatory_anchors": int(mapping["is_mandatory"].sum()),
        "protected_compact_anchors": int(mapping["is_protected_anchor"].sum()),
        "coverage_anchors": len(expected_coverage_rows),
        "coverage_anchors_retained": len(expected_coverage_rows - missing_coverage_rows),
        "union_candidates": len(selected_rows),
        "redundancy_removed": len(removed_to),
        "within_shot_redundancy_removed": len(removed_to) - len(adjacent_removed_to),
        "global_within_shot_redundancy_removed": len(global_within_removed_to),
        "adjacent_shot_redundancy_removed": len(adjacent_removed_to),
        "selected_keyframes": len(selected),
        "retained_ratio": len(selected) / len(mapping),
        "compression_from_samples": 1.0 - len(selected) / len(mapping),
        "minimum_selected_per_shot": int(shot_frame_counts.min()),
        "maximum_selected_per_shot": int(shot_frame_counts.max()),
        "parameters": {
            "k_cap": k_cap,
            "redundancy_cosine": redundancy_cosine,
            "temporal_window_frames": temporal_window_frames,
            "minimum_frames_per_shot": minimum_frames_per_shot,
            "protected_anchor_policy": protected_anchor_policy,
            "adjacent_shot_cosine": adjacent_shot_cosine,
            "global_within_shot_cosine": global_within_shot_cosine,
            "single_cluster_max_distance": single_cluster_max_distance,
        },
        "k_distribution": dict(sorted(Counter(x["k"] for x in shot_stats).items())),
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": {
            "selected_manifest": str(selected_path),
            "audit_shadow_manifest": str(audit_path),
            "selected_embeddings": str(compact_path),
        },
    }
    atomic_write_json(report, report_path)
    return report
