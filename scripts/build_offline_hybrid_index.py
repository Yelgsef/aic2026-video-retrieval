from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SO_ENCODER = "google_siglip2_so400m_patch14_384"
SOURCE_ENCODER = "google_siglip2_base_patch16_224"


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


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
        try:
            scenes = pd.read_parquet(scene_root / video_id / "scenes.parquet").sort_values("scene_index")
            index_dir = ROOT / "workspace" / "retrieval" / SO_ENCODER / f"{video_id}_final_from_{SOURCE_ENCODER}"
            metadata = pd.read_parquet(index_dir / "metadata.parquet").reset_index(drop=True)
            embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
            if len(metadata) != len(embeddings):
                raise ValueError("SO400M metadata/embedding row mismatch")
            for scene in scenes.itertuples(index=False):
                mask = (metadata.frame_id >= int(scene.start_frame)) & (metadata.frame_id < int(scene.end_frame_exclusive))
                positions = np.flatnonzero(mask.to_numpy())
                if not len(positions):
                    midpoint = (float(scene.start_time) + float(scene.end_time)) / 2
                    positions = np.array([int(np.argmin(np.abs(metadata.pts_time.to_numpy() - midpoint)))])
                pooled = normalize(np.asarray(embeddings[positions], dtype=np.float32).mean(axis=0))
                row_id = len(rows)
                for position in positions:
                    frame_rows.append({"frame_row": len(frame_rows), "scene_row": row_id,
                                       "video_id": video_id, "frame_id": int(metadata.iloc[position].frame_id),
                                       "pts_time": float(metadata.iloc[position].pts_time)})
                    frame_vectors.append(normalize(np.asarray(embeddings[position], dtype=np.float32)))
                rows.append({
                    "row_id": row_id,
                    "document_id": str(scene.scene_id),
                    "video_id": video_id,
                    "scene_index": int(scene.scene_index),
                    "start_frame": int(scene.start_frame),
                    "end_frame_exclusive": int(scene.end_frame_exclusive),
                    "start_time": float(scene.start_time),
                    "end_time": float(scene.end_time),
                    "shot_count": int(scene.shot_count),
                    "representative_frame_ids": [int(value) for value in scene.representative_frame_ids],
                    "visual_embedding_count": int(len(positions)),
                    "asr_text": str(scene.asr_exact or ""),
                    "asr_context": str(scene.asr_context or ""),
                    "retrieval_text": " ".join(filter(None, [str(scene.asr_exact or ""), str(scene.asr_context or "")])),
                })
                vectors.append(pooled)
        except (OSError, ValueError, KeyError) as exc:
            skipped[video_id] = str(exc)

    if not rows:
        raise RuntimeError("No scene with matching SO400M embeddings was available")
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(vectors).astype(np.float32)
    table = pd.DataFrame(rows)
    np.save(args.output / "scene_embeddings.npy", matrix)
    table.to_parquet(args.output / "scenes.parquet", index=False)
    image_matrix = np.stack(frame_vectors).astype(np.float32)
    np.save(args.output / "frame_embeddings.npy", image_matrix)
    pd.DataFrame(frame_rows).to_parquet(args.output / "frames.parquet", index=False)

    database = args.output / "search.sqlite3"
    temporary = database.with_suffix(".sqlite3.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE scenes(row_id INTEGER PRIMARY KEY, document_id TEXT UNIQUE, video_id TEXT, scene_index INTEGER, start_time REAL, end_time REAL, representative_frame_ids TEXT)")
    connection.execute("CREATE VIRTUAL TABLE scene_fts USING fts5(document_id UNINDEXED, asr_text, asr_context, retrieval_text, tokenize='unicode61 remove_diacritics 2')")
    connection.executemany("INSERT INTO scenes VALUES(?,?,?,?,?,?,?)", [
        (r["row_id"], r["document_id"], r["video_id"], r["scene_index"], r["start_time"], r["end_time"], json.dumps(r["representative_frame_ids"])) for r in rows
    ])
    connection.executemany("INSERT INTO scene_fts(rowid,document_id,asr_text,asr_context,retrieval_text) VALUES(?,?,?,?,?)", [
        (r["row_id"] + 1, r["document_id"], r["asr_text"], r["asr_context"], r["retrieval_text"]) for r in rows
    ])
    connection.commit()
    connection.close()
    temporary.replace(database)
    report = {"schema_version": "0.1.0", "videos": len(set(table.video_id)), "scenes": len(table),
              "scene_embedding_shape": list(matrix.shape), "frame_embedding_shape": list(image_matrix.shape),
              "skipped": skipped}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
