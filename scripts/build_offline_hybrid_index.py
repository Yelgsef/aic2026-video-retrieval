from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SO_ENCODER = "google_siglip2_so400m_patch14_384"
SOURCE_ENCODER = "google_siglip2_base_patch16_224"


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def character_ngrams(text: str, size: int = 3) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    compact = " ".join("".join(char if char.isalnum() else " " for char in text).split())
    return " ".join(sorted({compact[index:index + size].replace(" ", "_")
                            for index in range(max(0, len(compact) - size + 1))
                            if len(compact[index:index + size].strip()) == size}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the global offline scene-level hybrid index.")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / "offline_index")
    parser.add_argument("--video-id", action="append", help="Repeat to restrict the build; default uses all ready videos.")
    args = parser.parse_args()

    scene_root = ROOT / "workspace" / "semantic_scenes"
    video_ids = args.video_id or sorted(path.name for path in scene_root.iterdir() if path.is_dir())
    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    frame_rows: list[dict] = []
    frame_vectors: list[np.ndarray] = []
    skipped: dict[str, str] = {}
    for video_id in video_ids:
        video_rows: list[dict] = []
        video_vectors: list[np.ndarray] = []
        video_frame_rows: list[dict] = []
        video_frame_vectors: list[np.ndarray] = []
        try:
            scenes = pd.read_parquet(scene_root / video_id / "scenes.parquet").sort_values("scene_index")
            document_path = ROOT / "workspace" / "scene_documents" / video_id / "scenes.parquet"
            documents = pd.read_parquet(document_path).set_index("document_id") if document_path.exists() else None
            jsonl_path = ROOT / "workspace" / "scene_documents" / video_id / "scenes.jsonl"
            document_json = {}
            if jsonl_path.exists():
                with jsonl_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        payload = json.loads(line)
                        document_json[str(payload["document_id"])] = payload
            index_dir = ROOT / "workspace" / "retrieval" / SO_ENCODER / f"{video_id}_final_from_{SOURCE_ENCODER}"
            metadata = pd.read_parquet(index_dir / "metadata.parquet").reset_index(drop=True)
            embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
            if len(metadata) != len(embeddings):
                raise ValueError("SO400M metadata/embedding row mismatch")
            for scene in scenes.itertuples(index=False):
                document = documents.loc[str(scene.scene_id)] if documents is not None and str(scene.scene_id) in documents.index else None
                representative_frame_ids = [int(value) for value in scene.representative_frame_ids]
                ocr_evidence = []
                raw_document = document_json.get(str(scene.scene_id), {})
                for line in (raw_document.get("ocr_lines", [])
                             + raw_document.get("fullres_ocr_lines", [])):
                    if not isinstance(line, dict):
                        continue
                    if line.get("frame_id") is not None and line.get("frame_mapping_valid", True):
                        ocr_evidence.append({"text": str(line.get("text", "")),
                                             "frame_id": int(line["frame_id"])})
                mask = (metadata.frame_id >= int(scene.start_frame)) & (metadata.frame_id < int(scene.end_frame_exclusive))
                positions = np.flatnonzero(mask.to_numpy())
                fallback_used = not len(positions)
                if fallback_used:
                    midpoint = (float(scene.start_time) + float(scene.end_time)) / 2
                    positions = np.array([int(np.argmin(np.abs(metadata.pts_time.to_numpy() - midpoint)))])
                pooled = normalize(np.asarray(embeddings[positions], dtype=np.float32).mean(axis=0))
                row_id = len(rows) + len(video_rows)
                for position in positions:
                    video_frame_rows.append({
                        "frame_row": len(frame_rows) + len(video_frame_rows), "scene_row": row_id,
                        "video_id": video_id, "frame_id": int(metadata.iloc[position].frame_id),
                        "pts_time": float(metadata.iloc[position].pts_time),
                    })
                    video_frame_vectors.append(
                        normalize(np.asarray(embeddings[position], dtype=np.float32)))
                video_rows.append({
                    "row_id": row_id,
                    "document_id": str(scene.scene_id),
                    "video_id": video_id,
                    "scene_index": int(scene.scene_index),
                    "start_frame": int(scene.start_frame),
                    "end_frame_exclusive": int(scene.end_frame_exclusive),
                    "start_time": float(scene.start_time),
                    "end_time": float(scene.end_time),
                    "shot_count": int(scene.shot_count),
                    "representative_frame_ids": representative_frame_ids,
                    "ocr_evidence_json": json.dumps(ocr_evidence, ensure_ascii=False),
                    "visual_embedding_count": int(len(positions)),
                    "visual_fallback_used": fallback_used,
                    "asr_text": str(document.asr_text if document is not None else scene.asr_exact or ""),
                    "asr_context": str(document.asr_context if document is not None else scene.asr_context or ""),
                    "visual_summary": str(document.visual_summary if document is not None else ""),
                    "ocr_text": list(document.ocr_text) if document is not None else [],
                    "entities_text": " ".join(str(value) for field in ("people", "objects", "actions", "locations", "events", "search_keywords")
                                              for value in (document[field] if document is not None else [])),
                    "object_spatial_json": (json.dumps(document.object_spatial, ensure_ascii=False,
                                                         default=lambda value: value.tolist())
                                            if document is not None and "object_spatial" in document.index
                                            else "{}"),
                    "object_spatial_text": (str(document.object_spatial_text)
                                            if document is not None and "object_spatial_text" in document.index
                                            else ""),
                    "retrieval_text": str(document.retrieval_text if document is not None else " ".join(
                        filter(None, [str(scene.asr_exact or ""), str(scene.asr_context or "")]))),
                })
                video_vectors.append(pooled)
            # Commit this video's rows only after all its scenes are processed successfully.
            rows.extend(video_rows)
            vectors.extend(video_vectors)
            frame_rows.extend(video_frame_rows)
            frame_vectors.extend(video_frame_vectors)
        except (OSError, ValueError, KeyError) as exc:
            skipped[video_id] = str(exc)

    if not rows:
        raise RuntimeError("No scene with matching SO400M embeddings was available")
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(vectors).astype(np.float32)
    table = pd.DataFrame(rows)
    image_matrix = np.stack(frame_vectors).astype(np.float32)
    frame_table = pd.DataFrame(frame_rows)
    if len(table) != len(matrix) or len(frame_table) != len(image_matrix):
        raise ValueError("Index metadata/vector row mismatch before writing")
    if not np.array_equal(table.row_id.to_numpy(), np.arange(len(table))):
        raise ValueError("Scene row_id does not match scene vector order")
    if not np.array_equal(frame_table.frame_row.to_numpy(), np.arange(len(frame_table))):
        raise ValueError("Frame row_id does not match frame vector order")

    with tempfile.TemporaryDirectory(prefix=".offline_index_staging_", dir=args.output) as staging_name:
        staging = Path(staging_name)
        np.save(staging / "scene_embeddings.npy", matrix)
        table.to_parquet(staging / "scenes.parquet", index=False)
        np.save(staging / "frame_embeddings.npy", image_matrix)
        frame_table.to_parquet(staging / "frames.parquet", index=False)

        database = staging / "search.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE scenes(row_id INTEGER PRIMARY KEY, document_id TEXT UNIQUE, video_id TEXT, scene_index INTEGER, start_time REAL, end_time REAL, representative_frame_ids TEXT)")
        connection.execute("CREATE VIRTUAL TABLE scene_fts USING fts5(document_id UNINDEXED, asr_text, asr_context, visual_summary, ocr_text, entities_text, retrieval_text, tokenize='unicode61 remove_diacritics 2')")
        connection.execute("CREATE VIRTUAL TABLE fuzzy_fts USING fts5(document_id UNINDEXED, grams, tokenize='unicode61')")
        connection.executemany("INSERT INTO scenes VALUES(?,?,?,?,?,?,?)", [
            (r["row_id"], r["document_id"], r["video_id"], r["scene_index"], r["start_time"], r["end_time"], json.dumps(r["representative_frame_ids"])) for r in rows
        ])
        connection.executemany("INSERT INTO scene_fts(rowid,document_id,asr_text,asr_context,visual_summary,ocr_text,entities_text,retrieval_text) VALUES(?,?,?,?,?,?,?,?)", [
            (r["row_id"] + 1, r["document_id"], r["asr_text"], r["asr_context"], r["visual_summary"],
             " ".join(r["ocr_text"]), r["entities_text"] + " " + r["object_spatial_text"], r["retrieval_text"]) for r in rows
        ])
        connection.executemany("INSERT INTO fuzzy_fts(rowid,document_id,grams) VALUES(?,?,?)", [
            (r["row_id"] + 1, r["document_id"], character_ngrams(" ".join(r["ocr_text"]) + " " + r["entities_text"] + " " + r["retrieval_text"]))
            for r in rows
        ])
        connection.commit()
        connection.close()
        report = {"schema_version": "0.2.0", "videos": len(set(table.video_id)), "scenes": len(table),
                  "scene_embedding_shape": list(matrix.shape), "frame_embedding_shape": list(image_matrix.shape),
                  "visual_fallback_scenes": int(table.visual_fallback_used.sum()), "skipped": skipped}
        staged_report = staging / "report.json"
        staged_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        final_names = ("scene_embeddings.npy", "scenes.parquet", "frame_embeddings.npy",
                       "frames.parquet", "search.sqlite3")
        (args.output / "report.json").unlink(missing_ok=True)
        for name in final_names:
            (staging / name).replace(args.output / name)
        staged_report.replace(args.output / "report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
