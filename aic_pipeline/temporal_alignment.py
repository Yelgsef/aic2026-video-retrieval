from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor

from .query_understanding import build_query_plan


class TemporalAligner:
    def __init__(self, root: Path, device: str | None = None,
                 model_id: str = "google/siglip2-base-patch16-224") -> None:
        self.root = root.resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype, local_files_only=True).to(self.device).eval()

    def _event_vectors(self, events: list[str]) -> list[np.ndarray]:
        variant_groups = [list(build_query_plan(event).variants) for event in events]
        flat = [variant for group in variant_groups for variant in group]
        inputs = self.processor(text=flat, padding="max_length", truncation=True,
                                max_length=64, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vectors = self.model.get_text_features(**inputs).float()
            vectors = torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()
        output, offset = [], 0
        for group in variant_groups:
            group_vectors = vectors[offset:offset + len(group)]
            output.append(group_vectors)
            offset += len(group)
        return output

    def align(self, video_id: str, events: list[str]) -> dict[str, Any]:
        if len(events) < 2:
            raise ValueError("TRAKE requires at least two ordered events")
        feature_dir = self.root / "workspace" / "features" / "google_siglip2_base_patch16_224"
        metadata = pd.read_parquet(feature_dir / f"{video_id}.parquet").sort_values("embedding_row").reset_index(drop=True)
        embeddings = np.asarray(np.load(feature_dir / f"{video_id}.f16.npy", mmap_mode="r"), dtype=np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        groups = self._event_vectors(events)
        scores = np.stack([(embeddings @ group.T).max(axis=1) for group in groups])
        count, frames = scores.shape
        dp = np.full((count, frames), -np.inf, dtype=np.float32)
        parent = np.full((count, frames), -1, dtype=np.int64)
        dp[0] = scores[0]
        for event in range(1, count):
            best_score, best_index = -np.inf, -1
            for frame in range(frames):
                previous = frame - 1
                if previous >= 0 and dp[event - 1, previous] > best_score:
                    best_score, best_index = float(dp[event - 1, previous]), previous
                if best_index >= 0:
                    dp[event, frame] = scores[event, frame] + best_score
                    parent[event, frame] = best_index
        final = int(np.argmax(dp[-1]))
        positions = [final]
        for event in range(count - 1, 0, -1):
            positions.append(int(parent[event, positions[-1]]))
        positions.reverse()
        aligned = []
        for index, (event, position) in enumerate(zip(events, positions)):
            row = metadata.iloc[position]
            aligned.append({"event_index": index, "event": event, "frame_id": int(row.frame_id),
                            "pts_time": float(row.pts_time), "shot_id": int(row.shot_id),
                            "visual_score": float(scores[index, position])})
        return {"task": "TRAKE", "video_id": video_id, "events": aligned,
                "sequence_score": float(dp[-1, final] / count), "sampling_stride": 5,
                "maximum_sampling_offset_frames": 4}
