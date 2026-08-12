from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ID = "google/siglip2-base-patch16-224"
ENCODER_ID = "google_siglip2_base_patch16_224"


def encoder_id_from_model(model_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in model_id).strip("_")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero embedding")
    return vectors / norms


def save_index(
    output_dir: Path,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = l2_normalize(embeddings)
    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata row counts differ")
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    metadata.reset_index(drop=True).to_parquet(output_dir / "metadata.parquet", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    metadata = pd.read_parquet(index_dir / "metadata.parquet")
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    if len(embeddings) != len(metadata):
        raise ValueError("Corrupt index: embedding and metadata row counts differ")
    return embeddings, metadata, config


def exact_search(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    query_embedding: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = l2_normalize(np.asarray(query_embedding).reshape(1, -1))[0]
    if embeddings.shape[1] != query.shape[0]:
        raise ValueError("Query and image embedding dimensions differ")
    scores = np.asarray(embeddings @ query, dtype=np.float32)
    k = min(top_k, len(scores))
    candidates = np.argpartition(-scores, k - 1)[:k]
    order = candidates[np.argsort(-scores[candidates])]
    results = metadata.iloc[order].copy().reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, k + 1))
    results.insert(1, "score", scores[order])
    return results


def fuse_variant_scores(score_matrix: np.ndarray, method: str = "max") -> np.ndarray:
    """Fuse [query_variant, image] cosine scores into one score per image."""
    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("score_matrix must contain at least one query variant")
    if method == "max":
        return scores.max(axis=0)
    if method == "mean":
        return scores.mean(axis=0)
    if method == "zscore_max":
        means = scores.mean(axis=1, keepdims=True)
        stds = scores.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        return ((scores - means) / stds).max(axis=0)
    raise ValueError(f"Unsupported fusion method: {method}")


def rank_with_shot_dedup(
    scores: np.ndarray,
    metadata: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Rank frames while retaining only the best-scoring frame from each shot."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(scores) != len(metadata):
        raise ValueError("Score and metadata row counts differ")
    ranked = metadata.copy()
    ranked.insert(0, "score", np.asarray(scores, dtype=np.float32))
    ranked = ranked.sort_values("score", ascending=False, kind="stable")
    ranked = ranked.drop_duplicates(["video_id", "shot_id"], keep="first").head(top_k)
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked
