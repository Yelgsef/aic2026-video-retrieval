from __future__ import annotations

import json
import os
import sqlite3
import threading
import unicodedata
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, CLIPModel

from .query_understanding import build_query_plan
from .retrieval_fusion import field_coverage, robust_unit
from .submission import build_ranked_submissions
from .task_router import route_query


def char_ngrams(text: str, size: int = 3) -> list[str]:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    compact = " ".join("".join(char if char.isalnum() else " " for char in text).split())
    return sorted({compact[i:i + size].replace(" ", "_") for i in range(max(0, len(compact) - size + 1))
                   if len(compact[i:i + size].strip()) == size})


def intent_weights(intent: str) -> dict[str, float]:
    if intent == "ocr":
        return {"visual": 1.0, "openclip": 0.5, "lexical": 1.5, "fuzzy": 1.5, "semantic": 1.0, "reranker": 2.0}
    if intent == "asr":
        return {"visual": 0.75, "openclip": 0.25, "lexical": 1.75, "fuzzy": 0.5, "semantic": 1.5, "reranker": 2.0}
    return {"visual": 2.0, "openclip": 0.75, "lexical": 0.25, "fuzzy": 0.25, "semantic": 0.75, "reranker": 1.0}


class OfflineSearchEngine:
    def __init__(self, root: Path, index: Path | None = None, device: str | None = None) -> None:
        self.root = root.resolve()
        self.index = (index or self.root / "workspace" / "offline_index").resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.lock = threading.Lock()
        self.scenes = pd.read_parquet(self.index / "scenes.parquet")
        self.frames = pd.read_parquet(self.index / "frames.parquet")
        self.image_vectors = np.load(self.index / "frame_embeddings.npy", mmap_mode="r")
        openclip_vectors = self.index / "frame_embeddings_openclip.npy"
        self.openclip_vectors = np.load(openclip_vectors, mmap_mode="r") if openclip_vectors.exists() else None
        if self.openclip_vectors is not None and len(self.openclip_vectors) != len(self.frames):
            raise ValueError("OpenCLIP metadata/vector count mismatch")
        self.openclip_processor = None
        self.openclip = None
        self.text_vectors = np.load(self.index / "text_embeddings_bge_m3.npy", mmap_mode="r")
        self.field_text_vectors = {}
        for field in ("summary", "asr", "ocr_entities"):
            path = self.index / f"text_embeddings_bge_m3_{field}.npy"
            if path.exists():
                values = np.load(path, mmap_mode="r")
                if len(values) != len(self.scenes):
                    raise ValueError(f"{field} text vector count mismatch")
                self.field_text_vectors[field] = values
        if len(self.frames) != len(self.image_vectors) or len(self.scenes) != len(self.text_vectors):
            raise ValueError("Index metadata/vector count mismatch")

        siglip = self.root / "models" / "siglip2-so400m-patch14-384"
        siglip_id = str(siglip) if siglip.exists() else "google/siglip2-so400m-patch14-384"
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_id, local_files_only=True, use_fast=False)
        self.siglip = AutoModel.from_pretrained(siglip_id, dtype=self.dtype, local_files_only=True).to(self.device).eval()
        bge = self.root / "models" / "bge-m3"
        self.bge_tokenizer = AutoTokenizer.from_pretrained(str(bge), local_files_only=True)
        self.bge = AutoModel.from_pretrained(str(bge), dtype=self.dtype, local_files_only=True).to(self.device).eval()
        reranker = self.root / "models" / "bge-reranker-v2-m3"
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(str(reranker), local_files_only=True)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            str(reranker), dtype=self.dtype, local_files_only=True).to(self.device).eval()

    def _fts(self, query: str, table: str, limit: int) -> list[int]:
        if table == "scene_fts":
            stopwords = {"và", "với", "có", "một", "những", "nhiều", "của", "trong", "trên", "tại", "là"}
            tokens = [token.replace('"', "") for token in query.casefold().split()
                      if token.strip() and token not in stopwords]
        else:
            tokens = char_ngrams(query)
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        with sqlite3.connect(self.index / "search.sqlite3") as connection:
            return [int(row[0]) - 1 for row in connection.execute(
                f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table}) LIMIT ?",
                (expression, limit))]

    def _siglip_scores(self, variants: list[str]) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        inputs = self.siglip_processor(text=variants, padding="max_length", truncation=True, max_length=64,
                                       return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vectors = self.siglip.get_text_features(**inputs).float()
            vectors = torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()
        frame_scores = np.asarray((self.image_vectors @ vectors.T).max(axis=1), dtype=np.float32)
        scores = np.full(len(self.scenes), -np.inf, dtype=np.float32)
        best_frames = np.full(len(self.scenes), -1, dtype=np.int64)
        scene_candidates: list[list[tuple[float, int]]] = [[] for _ in range(len(self.scenes))]
        for frame_row, score in enumerate(frame_scores):
            scene_row = int(self.frames.iloc[frame_row].scene_row)
            frame_id = int(self.frames.iloc[frame_row].frame_id)
            scene_candidates[scene_row].append((float(score), frame_id))
            if score > scores[scene_row]:
                scores[scene_row] = score
                best_frames[scene_row] = frame_id
        top_frames = []
        for candidates in scene_candidates:
            selected = []
            for _, frame_id in sorted(candidates, reverse=True):
                if all(abs(frame_id - existing) >= 10 for existing in selected):
                    selected.append(frame_id)
                if len(selected) == 4:
                    break
            top_frames.append(selected)
        return scores, best_frames, top_frames

    def _bge_scores(self, query: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        inputs = self.bge_tokenizer([query], padding=True, truncation=True, max_length=512,
                                    return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vector = self.bge(**inputs).last_hidden_state[:, 0].float()
            vector = torch.nn.functional.normalize(vector, dim=-1).cpu().numpy()[0]
        primary = np.asarray(self.text_vectors @ vector, dtype=np.float32)
        fields = {name: np.asarray(values @ vector, dtype=np.float32)
                  for name, values in self.field_text_vectors.items()}
        return primary, fields

    def _openclip_scores(self, variants: list[str]) -> np.ndarray:
        if self.openclip_vectors is None:
            raise FileNotFoundError("OpenCLIP frame embeddings are not available")
        if self.openclip is None:
            model_dir = self.root / "models/openclip-vit-l14-laion2b"
            self.openclip_processor = AutoProcessor.from_pretrained(
                str(model_dir), local_files_only=True, use_fast=False)
            # Keep the independent model on CPU: the 8 GB GPU already hosts
            # SigLIP, BGE-M3 and the reranker. Text-only inference is small.
            self.openclip = CLIPModel.from_pretrained(str(model_dir), local_files_only=True).eval()
        inputs = self.openclip_processor(text=variants, padding=True, truncation=True,
                                         max_length=77, return_tensors="pt")
        with torch.inference_mode():
            vectors = self.openclip.get_text_features(**inputs).float()
            vectors = torch.nn.functional.normalize(vectors, dim=-1).numpy()
        frame_scores = np.asarray((self.openclip_vectors @ vectors.T).max(axis=1), dtype=np.float32)
        scene_scores = np.full(len(self.scenes), -np.inf, dtype=np.float32)
        for frame_row, score in enumerate(frame_scores):
            scene_row = int(self.frames.iloc[frame_row].scene_row)
            if score > scene_scores[scene_row]:
                scene_scores[scene_row] = score
        return scene_scores

    def _rerank(self, query: str, rows: list[int], batch_size: int) -> dict[int, float]:
        output: dict[int, float] = {}
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            pairs = [[query, str(self.scenes.iloc[row].retrieval_text)] for row in selected]
            inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, max_length=512,
                                              return_tensors="pt").to(self.device)
            with torch.inference_mode():
                values = torch.sigmoid(self.reranker(**inputs).logits.float().reshape(-1)).cpu().tolist()
            output.update({row: float(value) for row, value in zip(selected, values)})
        return output

    def search(self, query: str, top_k: int = 100, candidate_k: int = 300,
               rerank_k: int = 30, rerank_batch_size: int = 4,
               fusion_mode: str = "rrf", field_aware: bool = False,
               openclip: bool = False) -> dict[str, Any]:
        if not query.strip() or top_k <= 0:
            raise ValueError("query must be non-empty and top_k positive")
        with self.lock, torch.inference_mode():
            task = route_query(query)
            plan = build_query_plan(task.retrieval_query)
            weights = intent_weights(plan.intent)
            visual, best_frames, top_frames = self._siglip_scores(list(plan.variants))
            semantic, semantic_fields = self._bge_scores(plan.normalized)
            openclip_scores = self._openclip_scores(list(plan.variants)) if openclip else None
            if field_aware and semantic_fields:
                if plan.intent == "asr" and "asr" in semantic_fields:
                    semantic = np.maximum(semantic, semantic_fields["asr"])
                elif plan.intent == "ocr" and "ocr_entities" in semantic_fields:
                    semantic = np.maximum(semantic, semantic_fields["ocr_entities"])
                else:
                    semantic = np.maximum(semantic, semantic_fields.get("summary", semantic))
                    semantic = np.maximum(semantic, 0.92 * semantic_fields.get("ocr_entities", semantic))
            if fusion_mode not in {"rrf", "normalized"}:
                raise ValueError("fusion_mode must be rrf or normalized")
            orders = {
                "visual": np.argsort(-visual)[:candidate_k].tolist(),
                "semantic": np.argsort(-semantic)[:candidate_k].tolist(),
                "lexical": self._fts(plan.normalized, "scene_fts", candidate_k),
                "fuzzy": self._fts(plan.normalized, "fuzzy_fts", candidate_k),
            }
            if openclip_scores is not None:
                orders["openclip"] = np.argsort(-openclip_scores)[:candidate_k].tolist()
            fused: dict[int, float] = {}
            for branch, order in orders.items():
                for rank, row in enumerate(order, 1):
                    fused[row] = fused.get(row, 0.0) + weights[branch] / (60 + rank)
            rapid = {row: fuzz.WRatio(plan.normalized, str(self.scenes.iloc[row].retrieval_text)) for row in fused}
            for rank, row in enumerate(sorted(rapid, key=rapid.get, reverse=True), 1):
                fused[row] += weights["fuzzy"] / (60 + rank)
            coverage_scores: dict[int, float] = {}
            coverage_fields: dict[int, dict[str, float]] = {}
            if fusion_mode == "normalized":
                visual_unit, semantic_unit = robust_unit(visual), robust_unit(semantic)
                for row in fused:
                    coverage, fields = field_coverage(plan.normalized, self.scenes.iloc[row], plan.intent)
                    coverage_scores[row], coverage_fields[row] = coverage, fields
                    # RRF remains the robust backbone. Calibrated dense evidence and
                    # field coverage only break ties; they cannot remove a candidate.
                    fused[row] += (0.006 * weights["visual"] * float(visual_unit[row])
                                   + 0.006 * weights["semantic"] * float(semantic_unit[row])
                                   + 0.010 * coverage)
            preliminary = sorted(fused, key=fused.get, reverse=True)[:rerank_k]
            has_visual_text = any(bool(str(self.scenes.iloc[row].visual_summary).strip()) for row in preliminary)
            use_reranker = plan.intent != "visual" or has_visual_text
            reranker_scores = self._rerank(plan.normalized, preliminary, rerank_batch_size) if use_reranker else {}
            for rank, row in enumerate(sorted(reranker_scores, key=reranker_scores.get, reverse=True), 1):
                fused[row] += weights["reranker"] / (60 + rank)
            ranked = sorted(fused, key=fused.get, reverse=True)[:min(top_k, len(fused))]
            results = []
            for rank, row_id in enumerate(ranked, 1):
                row = self.scenes.iloc[row_id]
                results.append({"rank": rank, "fusion_score": fused[row_id], "visual_score": float(visual[row_id]),
                                "semantic_score": float(semantic[row_id]), "rapidfuzz_score": float(rapid[row_id]),
                                "openclip_score": (float(openclip_scores[row_id])
                                                   if openclip_scores is not None else None),
                                "semantic_field_scores": {name: float(values[row_id])
                                                          for name, values in semantic_fields.items()},
                                "field_coverage_score": coverage_scores.get(row_id),
                                "field_coverage": coverage_fields.get(row_id, {}),
                                "reranker_score": reranker_scores.get(row_id), "video_id": str(row.video_id),
                                "scene_index": int(row.scene_index), "start_time": float(row.start_time),
                                "end_time": float(row.end_time),
                                "start_frame": int(row.start_frame),
                                "end_frame_exclusive": int(row.end_frame_exclusive),
                                "best_frame_id": int(best_frames[row_id]),
                                "top_visual_frame_ids": top_frames[row_id],
                                "representative_frame_ids": [int(value) for value in row.representative_frame_ids],
                                "asr_text": "" if pd.isna(row.asr_text) else str(row.asr_text),
                                "scene_context": str(row.retrieval_text)[:600]})
            submissions = build_ranked_submissions(task.task, results, min(100, top_k)) if task.task != "TRAKE" else []
            return {"query": query, "task_plan": task.to_dict(), "query_plan": plan.to_dict(), "weights": weights,
                    "fusion_mode": fusion_mode, "field_aware": field_aware, "openclip_used": openclip,
                    "reranker_used": use_reranker, "results": results, "submissions": submissions,
                    "requires_temporal_alignment": task.task == "TRAKE"}

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "device": str(self.device), "scenes": len(self.scenes),
                "frames": len(self.frames), "image_dimension": int(self.image_vectors.shape[1]),
                "text_dimension": int(self.text_vectors.shape[1]),
                "semantic_fields": sorted(self.field_text_vectors),
                "openclip_available": self.openclip_vectors is not None,
                "cuda_memory_allocated": int(torch.cuda.memory_allocated()) if self.device.type == "cuda" else 0}
