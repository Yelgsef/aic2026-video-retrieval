from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor

from .query_understanding import build_query_plan
from .retrieval_fusion import robust_unit
from .temporal_reranker import motion_aware_scores


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

    def refine_kis_moment(self, video_id: str, query: str,
                          scene_windows: list[dict[str, Any]],
                          clip_scales: tuple[float, ...] = (0.8, 1.6, 3.2),
                          maximum_windows: int = 8) -> dict[str, Any]:
        """Choose a precise KIS frame inside a retrieved video.

        Scene retrieval is deliberately coarse. This method reuses the dense
        stride-5 SigLIP2 feature stream to resolve short actions and ordinal
        wording without decoding the video or making an external API call.
        """
        feature_dir = self.root / "workspace" / "features" / "google_siglip2_base_patch16_224"
        metadata = pd.read_parquet(feature_dir / f"{video_id}.parquet").sort_values(
            "embedding_row").reset_index(drop=True)
        embeddings = np.asarray(np.load(feature_dir / f"{video_id}.f16.npy", mmap_mode="r"),
                                dtype=np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        group = self._event_vectors([query])[0]
        frame_scores = (embeddings @ group.T).max(axis=1)[None, :]
        scores = self._fuse_short_clip_scores(
            frame_scores, embeddings, metadata.pts_time.to_numpy(), [group], clip_scales)[0]
        motion_scores, motion_gates = motion_aware_scores([query], embeddings, scores[None, :])
        if motion_gates[0]:
            scores = motion_scores[0]

        frame_ids = metadata.frame_id.to_numpy()
        eligible = np.zeros(len(metadata), dtype=bool)
        priors = np.zeros(len(metadata), dtype=np.float32)
        used_windows = []
        for rank, window in enumerate(scene_windows[:maximum_windows]):
            # Include context on both sides: ordinal actions often begin just
            # before the semantic scene selected by a static representative.
            span = max(1, int(window["end_frame_exclusive"]) - int(window["start_frame"]))
            margin = max(75, min(300, span // 2))
            mask = ((frame_ids >= int(window["start_frame"]) - margin)
                    & (frame_ids < int(window["end_frame_exclusive"]) + margin))
            eligible |= mask
            priors[mask] = np.maximum(priors[mask], 0.08 / (rank + 1))
            used_windows.append({"scene_index": int(window["scene_index"]),
                                 "start_frame": int(window["start_frame"]),
                                 "end_frame_exclusive": int(window["end_frame_exclusive"])})
        # Global peaks preserve recall when coarse retrieval selected the right
        # video but missed the precise scene.
        global_positions = np.argsort(-scores)[:100]
        eligible[global_positions] = True
        calibrated = robust_unit(scores) + priors
        positions = np.flatnonzero(eligible)
        if not len(positions):
            positions = np.arange(len(metadata))

        # Non-maximum suppression converts a run of near-identical stride-5
        # frames into one occurrence. Keep enough separation to distinguish
        # repeated gestures such as two claps or two scoops.
        peak_positions = []
        minimum_gap = 20
        for position in positions[np.argsort(-calibrated[positions])]:
            if all(abs(int(frame_ids[position]) - int(frame_ids[old])) >= minimum_gap
                   for old in peak_positions):
                peak_positions.append(int(position))
            if len(peak_positions) >= 40:
                break
        peak_positions.sort(key=lambda value: int(frame_ids[value]))
        folded = " ".join(query.casefold().split())
        ordinal_match = re.search(r"(?:lần\s+thứ|thứ)\s*(\d+)", folded)
        ordinal_words = {"một": 1, "nhất": 1, "hai": 2, "ba": 3,
                         "bốn": 4, "tư": 4, "năm": 5}
        ordinal_word_match = re.search(
            r"(?:lần\s+thứ|thứ)\s*(một|nhất|hai|ba|bốn|tư|năm)\b", folded)
        if ordinal_match:
            ordinal = max(1, int(ordinal_match.group(1)))
        elif ordinal_word_match:
            ordinal = ordinal_words[ordinal_word_match.group(1)]
        elif "đầu tiên" in folded or "lần đầu" in folded or "bắt đầu" in folded:
            ordinal = 1
        elif "cuối cùng" in folded:
            ordinal = -1
        else:
            ordinal = 0

        if ordinal and peak_positions:
            # Select the strongest temporal episode, then the requested peak
            # chronologically inside it. An episode gap of ten seconds avoids
            # treating unrelated occurrences elsewhere in a long video as the
            # first/second action.
            episode_gap = 300
            episodes: list[list[int]] = []
            for position in peak_positions:
                if not episodes or int(frame_ids[position]) - int(frame_ids[episodes[-1][-1]]) > episode_gap:
                    episodes.append([position])
                else:
                    episodes[-1].append(position)
            episode = max(episodes, key=lambda group_positions: float(
                max(calibrated[position] for position in group_positions)))
            strong = [position for position in episode
                      if calibrated[position] >= max(calibrated[p] for p in episode) - 0.18]
            if ordinal == -1:
                selected = strong[-1]
            else:
                selected = strong[min(ordinal - 1, len(strong) - 1)]
        else:
            selected = int(positions[np.argmax(calibrated[positions])])
        row = metadata.iloc[selected]
        return {"video_id": video_id, "frame_id": int(row.frame_id),
                "pts_time": float(row.pts_time), "shot_id": int(row.shot_id),
                "moment_score": float(calibrated[selected]), "ordinal": ordinal,
                "motion_reranking": bool(motion_gates[0]),
                "clip_scales": list(clip_scales), "used_windows": used_windows,
                "peak_frame_ids": [int(frame_ids[position]) for position in peak_positions]}

    def align(self, video_id: str, events: list[str],
              event_windows: list[list[dict[str, Any]]] | None = None,
              decoder: str = "windowed", short_clip: bool = False,
              clip_seconds: float = 2.0, motion_rerank: bool = False,
              clip_scales: tuple[float, ...] | None = None,
              global_peak_weight: float = 0.35) -> dict[str, Any]:
        if len(events) < 2:
            raise ValueError("TRAKE requires at least two ordered events")
        feature_dir = self.root / "workspace" / "features" / "google_siglip2_base_patch16_224"
        metadata = pd.read_parquet(feature_dir / f"{video_id}.parquet").sort_values("embedding_row").reset_index(drop=True)
        embeddings = np.asarray(np.load(feature_dir / f"{video_id}.f16.npy", mmap_mode="r"), dtype=np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        groups = self._event_vectors(events)
        scores = np.stack([(embeddings @ group.T).max(axis=1) for group in groups])
        if short_clip:
            scales = clip_scales or (clip_seconds,)
            scores = self._fuse_short_clip_scores(
                scores, embeddings, metadata.pts_time.to_numpy(), groups, scales)
        motion_gates = [False] * len(events)
        if motion_rerank:
            scores, motion_gates = motion_aware_scores(events, embeddings, scores)
        if decoder not in {"dense", "windowed"}:
            raise ValueError("decoder must be dense or windowed")
        # Coarse-to-fine fusion: scene retrieval supplies temporal priors, while
        # stride-5 SigLIP still chooses the exact frame inside/around each scene.
        if event_windows:
            frame_ids = metadata.frame_id.to_numpy()
            pts_times = metadata.pts_time.to_numpy()
            for event_index, windows in enumerate(event_windows):
                for rank, window in enumerate(windows[:5]):
                    if "start_frame" in window:
                        mask = ((frame_ids >= int(window["start_frame"]))
                                & (frame_ids < int(window["end_frame_exclusive"])))
                    else:
                        mask = ((pts_times >= float(window["start_time"]))
                                & (pts_times < float(window["end_time"])))
                    scores[event_index, mask] += 0.10 / (rank + 1)
        count, frames = scores.shape
        if decoder == "windowed" and event_windows:
            positions, sequence_score = self._windowed_decode(
                scores, metadata, event_windows, global_peak_weight=global_peak_weight)
        else:
            positions, sequence_score = self._dense_decode(scores)
        aligned = []
        for index, (event, position) in enumerate(zip(events, positions)):
            row = metadata.iloc[position]
            aligned.append({"event_index": index, "event": event, "frame_id": int(row.frame_id),
                            "pts_time": float(row.pts_time), "shot_id": int(row.shot_id),
                            "visual_score": float(scores[index, position])})
        return {"task": "TRAKE", "video_id": video_id, "events": aligned,
                "sequence_score": sequence_score, "sampling_stride": 5,
                "maximum_sampling_offset_frames": 4, "decoder": decoder,
                "short_clip_reranking": short_clip, "clip_seconds": clip_seconds if short_clip else None,
                "clip_scales": list(clip_scales or (clip_seconds,)) if short_clip else None,
                "motion_reranking": motion_rerank, "motion_event_gates": motion_gates,
                "global_peak_weight": global_peak_weight if decoder == "windowed" else None,
                "used_scene_priors": bool(event_windows)}

    @classmethod
    def _fuse_short_clip_scores(cls, frame_scores: np.ndarray, embeddings: np.ndarray,
                                times: np.ndarray, groups: list[np.ndarray],
                                scales: tuple[float, ...]) -> np.ndarray:
        """Fuse frame evidence with several temporal receptive fields.

        Each scale receives equal mass inside the 45% temporal budget. Keeping
        the original frame score at 55% prevents long windows from washing out
        short actions. A one-scale call is exactly the previous implementation.
        """
        cleaned = tuple(dict.fromkeys(float(value) for value in scales))
        if not cleaned or any(value <= 0 for value in cleaned):
            raise ValueError("clip scales must contain positive values")
        temporal = np.zeros_like(frame_scores, dtype=np.float32)
        for seconds in cleaned:
            pooled = cls._short_clip_embeddings(embeddings, times, seconds)
            temporal += np.stack([(pooled @ group.T).max(axis=1) for group in groups])
        temporal /= len(cleaned)
        return 0.55 * frame_scores + 0.45 * temporal

    @staticmethod
    def _short_clip_embeddings(embeddings: np.ndarray, times: np.ndarray,
                               seconds: float) -> np.ndarray:
        if seconds <= 0:
            raise ValueError("clip_seconds must be positive")
        output = np.empty_like(embeddings, dtype=np.float32)
        half = seconds / 2
        left = right = 0
        running = np.zeros(embeddings.shape[1], dtype=np.float64)
        for center, timestamp in enumerate(times):
            while right < len(times) and times[right] <= timestamp + half:
                running += embeddings[right]
                right += 1
            while left < len(times) and times[left] < timestamp - half:
                running -= embeddings[left]
                left += 1
            output[center] = running / max(1, right - left)
        output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
        return output

    @staticmethod
    def _dense_decode(scores: np.ndarray) -> tuple[list[int], float]:
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
        return positions, float(dp[-1, final] / count)

    @staticmethod
    def _windowed_decode(scores: np.ndarray, metadata: pd.DataFrame,
                         event_windows: list[list[dict[str, Any]]],
                         global_peak_weight: float = 0.35) -> tuple[list[int], float]:
        """Decode over diverse local peaks inside retrieved semantic windows."""
        if not 0 <= global_peak_weight <= 1:
            raise ValueError("global_peak_weight must be between zero and one")
        frame_ids = metadata.frame_id.to_numpy()
        candidates: list[list[tuple[int, float]]] = []
        for event_index, windows in enumerate(event_windows):
            calibrated = robust_unit(scores[event_index])
            event_candidates: dict[int, float] = {}
            window_scores = np.array([float(window.get("fusion_score", 0.0)) for window in windows[:8]],
                                     dtype=np.float32)
            window_units = robust_unit(window_scores)
            for rank, window in enumerate(windows[:8]):
                mask = ((frame_ids >= int(window["start_frame"]))
                        & (frame_ids < int(window["end_frame_exclusive"])))
                local = np.flatnonzero(mask)
                if not len(local):
                    continue
                selected: list[int] = []
                for position in local[np.argsort(-scores[event_index, local])]:
                    position = int(position)
                    if all(abs(int(frame_ids[position]) - int(frame_ids[old])) >= 10 for old in selected):
                        selected.append(position)
                    if len(selected) == 3:
                        break
                prior = 0.20 / (rank + 1) + 0.15 * float(window_units[rank])
                for position in selected:
                    value = float(calibrated[position]) + prior
                    event_candidates[position] = max(event_candidates.get(position, -np.inf), value)
            # Global peaks preserve recall when coarse retrieval misses the right scene.
            for position in np.argsort(-scores[event_index])[:20]:
                event_candidates[int(position)] = max(event_candidates.get(int(position), -np.inf),
                                                      global_peak_weight * float(calibrated[position]))
            candidates.append(sorted(event_candidates.items()))
        if any(not group for group in candidates):
            return TemporalAligner._dense_decode(scores)
        dp = [np.full(len(group), -np.inf, dtype=np.float32) for group in candidates]
        parent = [np.full(len(group), -1, dtype=np.int64) for group in candidates]
        dp[0] = np.array([score for _, score in candidates[0]], dtype=np.float32)
        for event_index in range(1, len(candidates)):
            for current_index, (current_position, current_score) in enumerate(candidates[event_index]):
                valid = [(previous_index, value) for previous_index, value in enumerate(dp[event_index - 1])
                         if candidates[event_index - 1][previous_index][0] < current_position and np.isfinite(value)]
                if valid:
                    previous_index, previous_score = max(valid, key=lambda item: item[1])
                    dp[event_index][current_index] = float(previous_score) + current_score
                    parent[event_index][current_index] = previous_index
        if not np.isfinite(dp[-1]).any():
            return TemporalAligner._dense_decode(scores)
        current = int(np.argmax(dp[-1]))
        positions = [candidates[-1][current][0]]
        for event_index in range(len(candidates) - 1, 0, -1):
            current = int(parent[event_index][current])
            positions.append(candidates[event_index - 1][current][0])
        positions.reverse()
        return positions, float(np.max(dp[-1]) / len(candidates))
