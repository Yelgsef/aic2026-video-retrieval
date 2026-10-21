from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymilvus import DataType, MilvusClient


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VectorSource:
    name: str
    filename: str
    metadata_filename: str
    id_column: str
    entity_kind: str


SOURCES = (
    VectorSource("scene_siglip", "scene_embeddings.npy", "scenes.parquet", "row_id", "scene"),
    VectorSource("bge_retrieval", "text_embeddings_bge_m3.npy", "scenes.parquet", "row_id", "scene"),
    VectorSource("bge_summary", "text_embeddings_bge_m3_summary.npy", "scenes.parquet", "row_id", "scene"),
    VectorSource("bge_asr", "text_embeddings_bge_m3_asr.npy", "scenes.parquet", "row_id", "scene"),
    VectorSource("bge_ocr_entities", "text_embeddings_bge_m3_ocr_entities.npy", "scenes.parquet", "row_id", "scene"),
    VectorSource("gemini_text", "text_embeddings_gemini.npy", "scenes.parquet", "row_id", "scene"),
    VectorSource("frame_siglip", "frame_embeddings.npy", "frames.parquet", "frame_row", "frame"),
    VectorSource("frame_openclip", "frame_embeddings_openclip.npy", "frames.parquet", "frame_row", "frame"),
)


def load_source(index: Path, source: VectorSource) -> tuple[np.ndarray, pd.DataFrame]:
    vector_path = index / source.filename
    metadata_path = index / source.metadata_filename
    if not vector_path.is_file():
        raise FileNotFoundError(vector_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    vectors = np.load(vector_path, mmap_mode="r")
    metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
    if vectors.ndim != 2 or len(vectors) != len(metadata):
        raise ValueError(f"{source.name}: vector/metadata row mismatch")
    if source.id_column not in metadata:
        raise ValueError(f"{source.name}: missing metadata key {source.id_column}")
    required = ({"video_id", "document_id", "scene_index", "start_time", "end_time"}
                if source.entity_kind == "scene" else
                {"video_id", "scene_row", "frame_id", "pts_time"})
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"{source.name}: metadata missing columns {missing}")
    ids = metadata[source.id_column].to_numpy()
    if not np.array_equal(ids, np.arange(len(metadata))):
        raise ValueError(f"{source.name}: {source.id_column} must match vector row order")
    if source.entity_kind == "frame":
        if (metadata.scene_row < 0).any() or (metadata.scene_row >= len(pd.read_parquet(index / "scenes.parquet"))).any():
            raise ValueError(f"{source.name}: scene_row contains out-of-range values")
        if source.name == "frame_openclip":
            keys_path = index / "openclip_frame_keys.parquet"
            if not keys_path.is_file():
                raise FileNotFoundError(keys_path)
            keys = pd.read_parquet(keys_path)
            expected = metadata[["video_id", "frame_id"]].reset_index(drop=True)
            if not {"video_id", "frame_id"}.issubset(keys.columns) or not keys[
                    ["video_id", "frame_id"]].reset_index(drop=True).equals(expected):
                raise ValueError("OpenCLIP frame keys do not match frames.parquet order")
    return vectors, metadata


def milvus_rows(source: VectorSource, metadata: pd.DataFrame,
                vectors: np.ndarray, start: int, stop: int) -> list[dict[str, Any]]:
    rows = []
    for position in range(start, stop):
        item = metadata.iloc[position]
        if source.entity_kind == "scene":
            scene_row = int(item.row_id)
            rows.append({
                "id": scene_row,
                "vector": np.asarray(vectors[position], dtype=np.float32).tolist(),
            })
        else:
            rows.append({
                "id": int(item.frame_row),
                "vector": np.asarray(vectors[position], dtype=np.float32).tolist(),
            })
    return rows


def exact_top_k(vectors: np.ndarray, query: np.ndarray, limit: int) -> set[int]:
    scores = np.asarray(vectors @ query, dtype=np.float32)
    selected = np.argpartition(scores, -limit)[-limit:]
    return set(int(value) for value in selected[np.argsort(-scores[selected])])


def ensure_finite(vectors: np.ndarray, batch_size: int, name: str) -> None:
    for start in range(0, len(vectors), batch_size):
        if not np.isfinite(vectors[start:start + batch_size]).all():
            raise ValueError(f"{name}: vectors contain NaN or infinity")


def input_fingerprint(index: Path, source: VectorSource) -> str:
    paths = [index / source.filename, index / source.metadata_filename]
    if source.name == "frame_openclip":
        paths.append(index / "openclip_frame_keys.parquet")
    facts = []
    for path in paths:
        stat = path.stat()
        facts.append(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(facts).encode()).hexdigest()


def verify_collection(client: MilvusClient, collection_name: str, source: VectorSource,
                      vectors: np.ndarray, metadata: pd.DataFrame,
                      top_k: int, sample_count: int) -> dict[str, float | int]:
    count = len(vectors)
    limit = min(top_k, count)
    sample_positions = np.unique(np.linspace(0, count - 1, min(sample_count, count), dtype=int))
    recalls = []
    self_hits = 0
    for position in sample_positions:
        query = np.asarray(vectors[position], dtype=np.float32)
        result = client.search(
            collection_name=collection_name,
            data=[query.tolist()],
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit)}},
            limit=limit,
        )[0]
        expected = exact_top_k(vectors, query, limit)
        actual = {int(hit["id"]) for hit in result}
        recalls.append(len(expected & actual) / len(expected))
        self_hits += int(int(metadata.iloc[position][source.id_column]) in actual)
    return {"samples": len(sample_positions), "top_k": limit,
            "mean_recall_at_k": float(np.mean(recalls)),
            "self_hit_rate": self_hits / len(sample_positions)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local Milvus Lite ANN index from offline index artifacts.")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace/offline_index")
    parser.add_argument("--uri", type=Path,
                        help="Milvus Lite database file; defaults to <index>/milvus_lite.db")
    parser.add_argument("--collection-prefix", default="aic")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--verify-top-k", type=int, default=10)
    parser.add_argument("--verify-samples", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true",
                        help="Drop existing collections with this prefix before rebuilding.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.hnsw_m < 4 or args.ef_construction < 8:
        parser.error("batch-size, hnsw-m and ef-construction must be positive and in supported ranges")
    if args.verify_top_k < 1 or args.verify_samples < 1:
        parser.error("verification top-k and sample count must be positive")

    index = args.index.resolve()
    uri = (args.uri or index / "milvus_lite.db").resolve()
    prefix = args.collection_prefix.strip().lower()
    if not prefix or not prefix[0].isalpha() or not prefix.replace("_", "").isalnum():
        parser.error("--collection-prefix must start with a letter and contain only letters, digits or underscores")
    uri.parent.mkdir(parents=True, exist_ok=True)

    selected_sources = [source for source in SOURCES if (index / source.filename).is_file()]
    required = {"scene_siglip", "bge_retrieval", "frame_siglip"}
    missing_required = sorted(required - {source.name for source in selected_sources})
    if missing_required:
        raise FileNotFoundError(f"Missing required embedding matrices: {missing_required}")
    prepared = {}
    for source in selected_sources:
        vectors, metadata = load_source(index, source)
        if len(vectors) == 0:
            raise ValueError(f"{source.name}: no vectors")
        ensure_finite(vectors, args.batch_size, source.name)
        prepared[source.name] = (vectors, metadata)

    client = MilvusClient(uri=str(uri))
    report: dict[str, Any] = {
        "schema_version": "0.1.0", "uri": str(uri), "index": str(index),
        "metric": "COSINE", "index_type": "HNSW",
        "hnsw": {"M": args.hnsw_m, "efConstruction": args.ef_construction},
        "collections": {},
    }
    try:
        collection_names = {f"{prefix}_{source.name}" for source in selected_sources}
        existing = set(client.list_collections()) & collection_names
        if existing and not args.overwrite:
            raise FileExistsError(
                f"Milvus collections already exist: {sorted(existing)}; pass --overwrite to rebuild them")
        for name in sorted(existing):
            client.drop_collection(name)

        for source in selected_sources:
            collection_name = f"{prefix}_{source.name}"
            vectors, metadata = prepared[source.name]
            dimension = int(vectors.shape[1])
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
            indexes = client.prepare_index_params()
            indexes.add_index(
                field_name="vector", index_type="HNSW", metric_type="COSINE",
                params={"M": args.hnsw_m, "efConstruction": args.ef_construction})
            client.create_collection(collection_name, schema=schema, index_params=indexes)

            inserted = 0
            for start in range(0, len(vectors), args.batch_size):
                stop = min(start + args.batch_size, len(vectors))
                batch = milvus_rows(source, metadata, vectors, start, stop)
                client.insert(collection_name=collection_name, data=batch)
                inserted += len(batch)
            client.flush(collection_name)
            verification = verify_collection(
                client, collection_name, source, vectors, metadata,
                args.verify_top_k, args.verify_samples)
            # Duplicate/equivalent vectors can make the exact source ID one of
            # several valid neighbors, so recall is the reliable ANN gate.
            if verification["mean_recall_at_k"] < 0.8:
                raise RuntimeError(f"{collection_name} ANN verification failed: {verification}")
            report["collections"][collection_name] = {
                "vectors": inserted, "dimension": dimension,
                "source": source.filename, "metadata": source.metadata_filename,
                "input_fingerprint": input_fingerprint(index, source),
                "verification": verification,
            }
            print(json.dumps({"collection": collection_name, **report["collections"][collection_name]}),
                  flush=True)
        (index / "milvus_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        client.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
