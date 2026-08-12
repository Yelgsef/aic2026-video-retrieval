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
    temporal_chunk_frames: int = 67,
    temporal_chunk_overlap: int = 10,
    max_medoids_per_chunk: int = 2,
    coverage_window_frames: int = 103,
    evidence_manifest_path: Path | None = None,
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
    if temporal_chunk_frames <= 0:
        raise ValueError("temporal_chunk_frames must be positive")
    if not 0 <= temporal_chunk_overlap < temporal_chunk_frames:
        raise ValueError("temporal_chunk_overlap must be in [0, temporal_chunk_frames)")
    if max_medoids_per_chunk not in {1, 2}:
        raise ValueError("max_medoids_per_chunk must be 1 or 2")
    if coverage_window_frames < 0:
        raise ValueError("coverage_window_frames must be non-negative")

    # Legacy fixed coverage anchors are deliberately not protected. Coverage
    # is now audited after redundancy and filled only for empty windows.
    coverage_anchors = pd.Series(False, index=mapping.index)
    if protected_anchor_policy == "none":
        mapping["is_protected_anchor"] = coverage_anchors
    elif protected_anchor_policy == "all":
        mapping["is_protected_anchor"] = mapping["is_mandatory"].astype(bool) | coverage_anchors
    else:
        mapping["is_protected_anchor"] = mapping["sampling_reason"].map(
            lambda value: bool(core_anchor_names.intersection(str(value).split("|")))
        ) | coverage_anchors
    mapping["is_coverage_anchor"] = coverage_anchors

    audit = mapping.copy()
    audit["chunk_ids"] = ""
    audit["chunk_k_max"] = 0
    audit["chunk_silhouette_max"] = 0.0
    chunk_memberships: dict[int, list[str]] = {int(row): [] for row in audit.index}
    selected_rows: set[int] = set()
    medoid_rows: set[int] = set()
    shot_stats: list[dict[str, Any]] = []
    chunk_stats: list[dict[str, Any]] = []
    chunk_step = temporal_chunk_frames - temporal_chunk_overlap
    for shot_id, shot in mapping.groupby("shot_id", sort=True):
        shot_positions = shot.index.to_numpy(dtype=np.int64)
        local_medoids: set[int] = set()
        shot_start = int(shot["shot_start_frame"].iloc[0])
        shot_end = int(shot["shot_end_frame_exclusive"].iloc[0])
        chunk_index = 0
        for chunk_start in range(shot_start, shot_end, chunk_step):
            chunk_end = min(chunk_start + temporal_chunk_frames, shot_end)
            chunk = shot[(shot.frame_id >= chunk_start) & (shot.frame_id < chunk_end)]
            if chunk.empty:
                continue
            positions = chunk.index.to_numpy(dtype=np.int64)
            chunk_vectors = vectors[positions]
            labels, centers, chosen_k, silhouette = _cluster_shot(
                chunk_vectors, seed=int(shot_id) * 1009 + chunk_index * 97 + 17,
                k_cap=max_medoids_per_chunk,
                single_cluster_max_distance=single_cluster_max_distance,
            )
            chunk_id = f"{int(shot_id)}:{chunk_index}"
            for row in positions:
                chunk_memberships[int(row)].append(chunk_id)
                audit.at[int(row), "chunk_k_max"] = max(
                    int(audit.at[int(row), "chunk_k_max"]), chosen_k)
                audit.at[int(row), "chunk_silhouette_max"] = max(
                    float(audit.at[int(row), "chunk_silhouette_max"]), silhouette)
            for cluster_id in range(chosen_k):
                members = np.flatnonzero(labels == cluster_id)
                if not len(members):
                    continue
                medoid_local = int(members[np.argmax(chunk_vectors[members] @ centers[cluster_id])])
                row = int(positions[medoid_local])
                local_medoids.add(row)
                medoid_rows.add(row)
            chunk_stats.append({"shot_id": int(shot_id), "chunk_id": chunk_id,
                                "start_frame": chunk_start, "end_frame_exclusive": chunk_end,
                                "samples": len(chunk), "k": chosen_k,
                                "silhouette": silhouette})
            chunk_index += 1
            if chunk_end == shot_end:
                break
        protected = set(
            shot_positions[shot["is_protected_anchor"].to_numpy(dtype=bool)].tolist()
        )
        candidates = protected | local_medoids
        selected_rows.update(candidates)
        shot_stats.append(
            {"shot_id": int(shot_id), "samples": len(shot),
             "chunks": chunk_index, "candidates": len(candidates)}
        )

    audit["chunk_ids"] = ["|".join(chunk_memberships[int(row)]) for row in audit.index]
    # Compatibility aliases for downstream audit notebooks.
    audit["shot_k"] = audit["chunk_k_max"]
    audit["shot_silhouette"] = audit["chunk_silhouette_max"]
    audit["is_cluster_medoid"] = audit.index.isin(medoid_rows)
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

    # Post-selection coverage: add one central sampled fallback only when an
    # aligned global window has no retained frame.
    coverage_fallback_rows: set[int] = set()
    if coverage_window_frames:
        video_end = int(mapping["shot_end_frame_exclusive"].max())
        for window_start in range(0, video_end, coverage_window_frames):
            window_end = min(window_start + coverage_window_frames, video_end)
            if any(window_start <= int(audit.at[row, "frame_id"]) < window_end
                   for row in final_rows):
                continue
            candidates = audit[(audit.frame_id >= window_start) & (audit.frame_id < window_end)]
            if candidates.empty:
                continue
            center = (window_start + window_end - 1) / 2.0
            row = int((candidates.frame_id.astype(float) - center).abs().idxmin())
            final_rows.add(row)
            coverage_fallback_rows.add(row)
            removed_to.pop(row, None)

    # Optional pre-filtered evidence manifest. Producers should mark only
    # discriminative OCR/object novelty, not every common detection.
    evidence_rows: set[int] = set()
    if evidence_manifest_path is not None and evidence_manifest_path.exists():
        evidence = pd.read_parquet(evidence_manifest_path)
        if "frame_id" not in evidence:
            raise ValueError("Evidence manifest must contain frame_id")
        enabled = pd.Series(False, index=evidence.index)
        for column in ("keep", "ocr_evidence", "object_evidence"):
            if column in evidence:
                enabled |= evidence[column].fillna(False).astype(bool)
        wanted = set(evidence.loc[enabled, "frame_id"].astype(int))
        evidence_rows = set(audit.index[audit.frame_id.astype(int).isin(wanted)].astype(int))
        final_rows.update(evidence_rows)
        for row in evidence_rows:
            removed_to.pop(row, None)

    audit["selected"] = audit.index.isin(final_rows)
    audit["is_coverage_fallback"] = audit.index.isin(coverage_fallback_rows)
    audit["is_evidence_frame"] = audit.index.isin(evidence_rows)
    audit["redundant_to_embedding_row"] = pd.Series(removed_to, dtype="Int64")
    audit.loc[audit.index.isin(removed_to), "selection_stage"] = "redundancy_removed"
    audit.loc[audit.index.isin(adjacent_removed_to), "selection_stage"] = "adjacent_shot_removed"
    audit.loc[audit.index.isin(global_within_removed_to), "selection_stage"] = "global_within_shot_removed"
    audit.loc[audit.index.isin(coverage_fallback_rows), "selection_stage"] = "coverage_fallback"
    audit.loc[audit.index.isin(evidence_rows), "selection_stage"] = "ocr_object_evidence"
    audit = audit.reset_index(drop=True)

    selected = audit[audit["selected"]].sort_values("frame_id").copy()
    selected.insert(0, "keyframe_row", np.arange(len(selected), dtype=np.int64))
    selected["selection_reasons"] = selected.apply(
        lambda row: "|".join(
            reason for reason, enabled in (
                ("protected_anchor", bool(row["is_protected_anchor"])),
                ("coverage_anchor", bool(row["is_coverage_anchor"])),
                ("coverage_fallback", bool(row["is_coverage_fallback"])),
                ("ocr_object_evidence", bool(row["is_evidence_frame"])),
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
        "pipeline_version": "0.2.0",
        "video_id": video_id,
        "encoder_id": encoder_id,
        "input_samples": len(mapping),
        "shot_count": int(mapping["shot_id"].nunique()),
        "cluster_medoids": len(medoid_rows),
        "sampled_mandatory_anchors": int(mapping["is_mandatory"].sum()),
        "protected_compact_anchors": int(mapping["is_protected_anchor"].sum()),
        "coverage_fallbacks": len(coverage_fallback_rows),
        "evidence_frames": len(evidence_rows),
        "temporal_chunks": len(chunk_stats),
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
            "temporal_chunk_frames": temporal_chunk_frames,
            "temporal_chunk_overlap": temporal_chunk_overlap,
            "max_medoids_per_chunk": max_medoids_per_chunk,
            "coverage_window_frames": coverage_window_frames,
            "evidence_manifest_path": str(evidence_manifest_path) if evidence_manifest_path else None,
        },
        "k_distribution": dict(sorted(Counter(x["k"] for x in chunk_stats).items())),
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": {
            "selected_manifest": str(selected_path),
            "audit_shadow_manifest": str(audit_path),
            "selected_embeddings": str(compact_path),
        },
    }
    atomic_write_json(report, report_path)
    return report
