from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor

from .query_understanding import build_query_plan
from .retrieval_fusion import robust_unit


class XClipWindowReranker:
    """Candidate-only video-text reranker for TRAKE scene windows."""

    def __init__(self, root: Path, model_path: Path | None = None,
                 device: str | None = None, frames_per_clip: int = 8) -> None:
        self.root = root.resolve()
        self.model_path = (model_path or self.root / "models/xclip-base-patch32").resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.frames_per_clip = frames_per_clip
        self.manifest = pd.read_parquet(
            self.root / "workspace/manifests/videos.parquet").set_index("video_id")
        self.processor = XCLIPProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = XCLIPModel.from_pretrained(
            str(self.model_path), dtype=self.dtype, local_files_only=True).to(self.device).eval()

    def _decode_window(self, capture: cv2.VideoCapture, start: int, end: int) -> list[Image.Image]:
        last = max(start, end - 1)
        frame_ids = np.linspace(start, last, self.frames_per_clip).round().astype(int)
        images = []
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode X-CLIP frame {frame_id}")
            images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return images

    def rerank(self, video_id: str, events: list[str], windows: list[list[dict[str, Any]]],
               weight: float = 0.25, maximum_windows: int = 5,
               batch_size: int = 4) -> list[list[dict[str, Any]]]:
        if not 0 <= weight <= 1:
            raise ValueError("X-CLIP weight must be between zero and one")
        capture = cv2.VideoCapture(str(self.manifest.loc[video_id].video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video {video_id}")
        clips, owners = [], []
        try:
            for event_index, group in enumerate(windows):
                for window_index, window in enumerate(group[:maximum_windows]):
                    clips.append(self._decode_window(
                        capture, int(window["start_frame"]), int(window["end_frame_exclusive"])))
                    owners.append((event_index, window_index))
        finally:
            capture.release()
        # X-CLIP is English-centric. Use the most translated/gloss-heavy variant
        # available while retaining the raw event when no gloss exists.
        texts = [build_query_plan(event).variants[-1] for event in events]
        raw_scores = np.zeros(len(clips), dtype=np.float32)
        for offset in range(0, len(clips), batch_size):
            batch = clips[offset:offset + batch_size]
            inputs = self.processor(text=texts, images=batch, padding=True, truncation=True,
                                    return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
            with torch.inference_mode():
                logits = self.model(**inputs).logits_per_video.float().cpu().numpy()
            for local_index in range(len(batch)):
                event_index, _ = owners[offset + local_index]
                raw_scores[offset + local_index] = logits[local_index, event_index]
        output = [[dict(window) for window in group] for group in windows]
        for event_index, group in enumerate(output):
            indices = [index for index, owner in enumerate(owners) if owner[0] == event_index]
            if not indices:
                continue
            visual_units = robust_unit(raw_scores[indices])
            semantic_units = robust_unit(np.array(
                [float(group[owners[index][1]].get("fusion_score", 0.0)) for index in indices],
                dtype=np.float32))
            for local, index in enumerate(indices):
                _, window_index = owners[index]
                group[window_index]["xclip_score"] = float(raw_scores[index])
                group[window_index]["fusion_score"] = float(
                    (1 - weight) * semantic_units[local] + weight * visual_units[local])
            group[:maximum_windows] = sorted(
                group[:maximum_windows], key=lambda row: float(row["fusion_score"]), reverse=True)
        return output
