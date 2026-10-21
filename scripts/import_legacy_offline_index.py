from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VECTOR_FILES = (
    "scene_embeddings.npy",
    "frame_embeddings.npy",
    "frame_embeddings_openclip.npy",
    "text_embeddings_bge_m3.npy",
    "text_embeddings_bge_m3_summary.npy",
    "text_embeddings_bge_m3_asr.npy",
    "text_embeddings_bge_m3_ocr_entities.npy",
)
SCENE_COLUMNS = {
    "row_id", "document_id", "video_id", "scene_index", "start_frame",
    "end_frame_exclusive", "start_time", "end_time", "representative_frame_ids",
    "ocr_evidence_json", "asr_text", "asr_context", "visual_summary", "ocr_text",
    "entities_text", "object_spatial_json", "object_spatial_text", "retrieval_text",
}
FRAME_COLUMNS = {"frame_row", "scene_row", "video_id", "frame_id", "pts_time"}


def _text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, (list, tuple, np.ndarray)):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    return str(value)


def _list_text(value: object) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(_text(item) for item in value)
    return _text(value)


def character_ngrams(text: str, size: int = 3) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    compact = " ".join("".join(char if char.isalnum() else " " for char in text).split())
    return " ".join(sorted({compact[index:index + size].replace(" ", "_")
                            for index in range(max(0, len(compact) - size + 1))
                            if len(compact[index:index + size].strip()) == size}))


def validate_source(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    scenes = pd.read_parquet(source / "scenes.parquet").reset_index(drop=True)
    frames = pd.read_parquet(source / "frames.parquet").reset_index(drop=True)
    missing_scenes = SCENE_COLUMNS - set(scenes.columns)
    missing_frames = FRAME_COLUMNS - set(frames.columns)
    if missing_scenes or missing_frames:
        raise ValueError(f"Missing Parquet columns: scenes={sorted(missing_scenes)}, frames={sorted(missing_frames)}")
    if not np.array_equal(scenes.row_id.to_numpy(), np.arange(len(scenes))):
        raise ValueError("scenes.row_id must match scene vector row order")
    if not np.array_equal(frames.frame_row.to_numpy(), np.arange(len(frames))):
        raise ValueError("frames.frame_row must match frame vector row order")
    if frames.scene_row.isna().any() or (frames.scene_row < 0).any() or (frames.scene_row >= len(scenes)).any():
        raise ValueError("frames.scene_row contains invalid scene references")
    if scenes[["video_id", "document_id"]].isna().any().any() or scenes.document_id.duplicated().any():
        raise ValueError("scenes has missing or duplicate document identifiers")

    expected_rows = {
        "scene_embeddings.npy": len(scenes),
        "frame_embeddings.npy": len(frames),
        "frame_embeddings_openclip.npy": len(frames),
        "text_embeddings_bge_m3.npy": len(scenes),
        "text_embeddings_bge_m3_summary.npy": len(scenes),
        "text_embeddings_bge_m3_asr.npy": len(scenes),
        "text_embeddings_bge_m3_ocr_entities.npy": len(scenes),
    }
    shapes: dict[str, list[int]] = {}
    for filename, expected in expected_rows.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        vectors = np.load(path, mmap_mode="r")
        if vectors.ndim != 2 or len(vectors) != expected:
            raise ValueError(f"{filename}: expected {expected} rows, got shape {vectors.shape}")
        for start in range(0, len(vectors), 8192):
            if not np.isfinite(vectors[start:start + 8192]).all():
                raise ValueError(f"{filename}: contains NaN or infinity")
        shapes[filename] = list(vectors.shape)
    return scenes, frames, shapes


def build_search_database(path: Path, scenes: pd.DataFrame) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE scenes(row_id INTEGER PRIMARY KEY, document_id TEXT UNIQUE, video_id TEXT, scene_index INTEGER, start_time REAL, end_time REAL, representative_frame_ids TEXT)")
        connection.execute("CREATE VIRTUAL TABLE scene_fts USING fts5(document_id UNINDEXED, asr_text, asr_context, visual_summary, ocr_text, entities_text, retrieval_text, tokenize='unicode61 remove_diacritics 2')")
        connection.execute("CREATE VIRTUAL TABLE fuzzy_fts USING fts5(document_id UNINDEXED, grams, tokenize='unicode61')")

        scene_rows, fts_rows, fuzzy_rows = [], [], []
        for scene in scenes.itertuples(index=False):
            row_id = int(scene.row_id)
            document_id = str(scene.document_id)
            ocr_text = _list_text(scene.ocr_text)
            entities_text = _text(scene.entities_text) + " " + _text(scene.object_spatial_text)
            retrieval_text = _text(scene.retrieval_text)
            representative_frame_ids = [int(frame_id) for frame_id in scene.representative_frame_ids]
            scene_rows.append((row_id, document_id, str(scene.video_id), int(scene.scene_index),
                               float(scene.start_time), float(scene.end_time),
                               json.dumps(representative_frame_ids, ensure_ascii=False)))
            fts_rows.append((row_id + 1, document_id, _text(scene.asr_text), _text(scene.asr_context),
                             _text(scene.visual_summary), ocr_text, entities_text, retrieval_text))
            fuzzy_source = " ".join((ocr_text, _text(scene.entities_text), retrieval_text))
            fuzzy_rows.append((row_id + 1, document_id, character_ngrams(fuzzy_source)))
        connection.executemany("INSERT INTO scenes VALUES(?,?,?,?,?,?,?)", scene_rows)
        connection.executemany(
            "INSERT INTO scene_fts(rowid,document_id,asr_text,asr_context,visual_summary,ocr_text,entities_text,retrieval_text) VALUES(?,?,?,?,?,?,?,?)",
            fts_rows)
        connection.executemany("INSERT INTO fuzzy_fts(rowid,document_id,grams) VALUES(?,?,?)", fuzzy_rows)
        connection.commit()
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import only serving metadata and vector sources from a legacy offline index dump.")
    parser.add_argument("--source", type=Path, required=True,
                        help="Legacy offline_index_all_873 dump directory.")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/offline_index_all_873")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Source directory does not exist: {source}")
    if source == output or source in output.parents:
        parser.error("Output must not be the source directory or inside it")
    if output.exists():
        parser.error(f"Output already exists; refusing to overwrite: {output}")

    scenes, frames, shapes = validate_source(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.import_", dir=output.parent))
    try:
        selected_files = ("scenes.parquet", "frames.parquet", *VECTOR_FILES)
        for filename in selected_files:
            shutil.copy2(source / filename, staging / filename)

        frames[["video_id", "frame_id"]].to_parquet(
            staging / "openclip_frame_keys.parquet", index=False)
        build_search_database(staging / "search.sqlite3", scenes)
        report = {
            "schema_version": "1.0.0",
            "source": str(source),
            "videos": int(scenes.video_id.nunique()),
            "scenes": len(scenes),
            "frames": len(frames),
            "vector_shapes": shapes,
            "copied_files": list(selected_files),
            "generated_files": ["openclip_frame_keys.parquet", "search.sqlite3"],
            "omitted_files": ["scenes.parquet.before-cloud-20260828-1827.bak",
                              "query_transformer_cache.sqlite3"],
            "milvus": "Build separately with scripts/build_milvus_index.py.",
        }
        (staging / "import_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
