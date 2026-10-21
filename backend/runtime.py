from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from backend.config import Settings
from aic_pipeline.offline_search_engine import OfflineSearchEngine


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = OfflineSearchEngine(
            settings.root, settings.index, settings.device,
            settings.milvus_uri, settings.collection_prefix,
        )
        if settings.warmup:
            self.engine.search("người đang hoạt động", top_k=1, candidate_k=20,
                               rerank_k=4, openclip=self.engine.openclip_available,
                               query_transformer=False)
        self.scene_rows = {
            (str(row.video_id), int(row.scene_index)): index
            for index, row in enumerate(self.engine.scenes.itertuples(index=False))
        }
        manifest_path = settings.root / "workspace/manifests/videos.parquet"
        manifest = pd.read_parquet(
            manifest_path, columns=["video_id", "video_path", "fps", "frame_count", "duration_seconds"])
        self.video_paths = {
            str(row.video_id): Path(row.video_path).expanduser().resolve()
            for row in manifest.itertuples(index=False)
        }
        self.video_metadata = manifest.set_index("video_id")
        settings.frame_cache.mkdir(parents=True, exist_ok=True)

    def resolve_video(self, video_id: str) -> Path:
        path = self.video_paths.get(video_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Unknown or unavailable video_id: {video_id}")
        return path

    def materialize_candidate(self, item: dict[str, Any], rank: int | None = None) -> dict[str, Any]:
        """Join a ranked video/frame hit to its scene metadata for the UI."""
        video_id = str(item["video_id"])
        scene_index = int(item["scene_index"])
        scene_row = self.scene_rows.get((video_id, scene_index))
        if scene_row is None:
            raise ValueError(f"Scene {scene_index} is not indexed for {video_id}")
        scene = self.engine.scenes.iloc[scene_row]
        frame_id = int(item.get("best_frame_id", item.get("frame_id", 0)))
        return {
            "rank": rank if rank is not None else item.get("rank"),
            "video_id": video_id, "frame_id": frame_id, "best_frame_id": frame_id,
            "score": float(item.get("fusion_score", item.get("score", 0.0))),
            "fusion_score": float(item.get("fusion_score", item.get("score", 0.0))),
            "scene_index": scene_index, "start_time": float(scene.start_time),
            "end_time": float(scene.end_time), "start_frame": int(scene.start_frame),
            "end_frame_exclusive": int(scene.end_frame_exclusive),
            "scene_context": str(scene.retrieval_text)[:600],
            "asr_text": "" if pd.isna(scene.asr_text) else str(scene.asr_text),
            "ocr_text": ([str(value) for value in scene.ocr_text]
                         if isinstance(scene.ocr_text, (list, tuple, np.ndarray)) else []),
            "top_visual_frame_ids": [int(value) for value in item.get("top_visual_frame_ids", [])],
            "representative_frame_ids": [int(value) for value in scene.representative_frame_ids],
        }

    def close(self) -> None:
        self.engine.close()


def public_search(runtime: Runtime, data: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in data.items() if key not in {"results", "submissions"}}
    output["query_transformer_used"] = bool(data.get("query_transformer_used", False))
    output["retrieval_backend"] = "milvus_lite"
    results, seen = [], set()
    for item in data.get("results", []):
        candidate = runtime.materialize_candidate(item)
        key = (candidate["video_id"], candidate["frame_id"])
        if key in seen:
            continue
        seen.add(key)
        candidate["rank"] = len(results) + 1
        results.append(candidate)
    output["results"] = results
    output["submissions"] = data.get("submissions", [])
    return output
