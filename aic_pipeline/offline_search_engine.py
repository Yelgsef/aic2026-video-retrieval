from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from pymilvus import MilvusClient
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, CLIPModel

from .query_understanding import build_query_plan
from .query_transformer import (LocalQueryTransformer, QueryTransformerError,
                                QueryTransformerUnavailable)
from .object_spatial import structured_object_scores
from .fusion_profiles import accepted_profile
from .gemini_text_retrieval import GeminiTextServices
from .retrieval_fusion import aggregate_scene_scores, field_coverage, robust_unit
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
        return {"visual": 1.0, "openclip": 0.5, "lexical": 1.5, "fuzzy": 1.5,
                "semantic": 1.0, "text_first": 1.25, "object": 0.5, "reranker": 2.0}
    if intent == "asr":
        return {"visual": 0.75, "openclip": 0.25, "lexical": 1.75, "fuzzy": 0.5,
                "semantic": 1.5, "text_first": 1.5, "object": 0.25, "reranker": 2.0}
    return {"visual": 2.0, "openclip": 0.75, "lexical": 0.25, "fuzzy": 0.25,
            "semantic": 0.75, "text_first": 1.0, "object": 1.25, "reranker": 1.0}


class OfflineSearchEngine:
    def __init__(self, root: Path, index: Path | None = None, device: str | None = None,
                 milvus_uri: Path | None = None, collection_prefix: str = "aic") -> None:
        self.root = root.resolve()
        self.index = (index or self.root / "workspace" / "offline_index").resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.lock = threading.Lock()
        self.scenes = pd.read_parquet(self.index / "scenes.parquet")
        self.frames = pd.read_parquet(self.index / "frames.parquet")
        self.milvus_uri = (milvus_uri or self.index / "milvus_lite.db").resolve()
        if not self.milvus_uri.exists():
            raise FileNotFoundError(
                f"Milvus Lite index not found at {self.milvus_uri}; run scripts/build_milvus_index.py")
        self.collection_prefix = collection_prefix.strip().lower()
        self.milvus = MilvusClient(uri=str(self.milvus_uri))
        available = set(self.milvus.list_collections())
        self.milvus_collections = {
            source: f"{self.collection_prefix}_{source}"
            for source in ("scene_siglip", "frame_siglip", "bge_retrieval", "bge_summary",
                           "bge_asr", "bge_ocr_entities", "frame_openclip", "gemini_text")
            if f"{self.collection_prefix}_{source}" in available
        }
        required_collections = {"frame_siglip", "bge_retrieval"}
        missing_collections = required_collections - set(self.milvus_collections)
        if missing_collections:
            self.milvus.close()
            raise ValueError(f"Milvus index is missing required collections: {sorted(missing_collections)}")
        for collection in self.milvus_collections.values():
            self.milvus.load_collection(collection)
        self.openclip_available = "frame_openclip" in self.milvus_collections
        self.vector_dimensions = self._load_milvus_dimensions()
        self.openclip_processor = None
        self.openclip = None
        self.gemini_text_available = "gemini_text" in self.milvus_collections
        self.gemini_text_services: GeminiTextServices | bool | None = None
        self._exact_ocr_cache: dict[str, dict[int, int]] = {}
        self.query_transformer = LocalQueryTransformer(self.root, self.index)
        self.field_text_collections = {
            name: name for name in self.milvus_collections
            if name in {"bge_summary", "bge_asr", "bge_ocr_entities"}
        }

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

    def _load_milvus_dimensions(self) -> dict[str, int]:
        report_path = self.index / "milvus_report.json"
        if not report_path.is_file():
            self.milvus.close()
            raise FileNotFoundError(f"Milvus report missing at {report_path}; rebuild the Milvus index")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_files = {
            "scene_siglip": ("scene_embeddings.npy", "scenes.parquet"),
            "frame_siglip": ("frame_embeddings.npy", "frames.parquet"),
            "bge_retrieval": ("text_embeddings_bge_m3.npy", "scenes.parquet"),
            "bge_summary": ("text_embeddings_bge_m3_summary.npy", "scenes.parquet"),
            "bge_asr": ("text_embeddings_bge_m3_asr.npy", "scenes.parquet"),
            "bge_ocr_entities": ("text_embeddings_bge_m3_ocr_entities.npy", "scenes.parquet"),
            "frame_openclip": ("frame_embeddings_openclip.npy", "frames.parquet",
                                "openclip_frame_keys.parquet"),
            "gemini_text": ("text_embeddings_gemini.npy", "scenes.parquet"),
        }
        dimensions = {}
        for source, collection in self.milvus_collections.items():
            metadata = report.get("collections", {}).get(collection, {})
            expected_rows = len(self.frames) if source.startswith("frame_") else len(self.scenes)
            if int(metadata.get("vectors", -1)) != expected_rows:
                self.milvus.close()
                raise ValueError(f"Milvus collection {collection} has stale row count")
            facts = []
            for filename in source_files[source]:
                path = self.index / filename
                if not path.is_file():
                    self.milvus.close()
                    raise FileNotFoundError(f"Milvus source artifact missing: {path}")
                stat = path.stat()
                facts.append(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}")
            fingerprint = hashlib.sha256("\n".join(facts).encode()).hexdigest()
            if metadata.get("input_fingerprint") != fingerprint:
                self.milvus.close()
                raise ValueError(f"Milvus collection {collection} is stale; rebuild the Milvus index")
            dimension = int(metadata.get("dimension", 0))
            if dimension < 1:
                self.milvus.close()
                raise ValueError(f"Milvus report is missing dimensions for {collection}")
            dimensions[source] = dimension
        return dimensions

    def _milvus_search(self, source: str, queries: np.ndarray,
                       limit: int) -> list[list[dict[str, Any]]]:
        collection = self.milvus_collections.get(source)
        if collection is None:
            return [[] for _ in queries]
        entity_count = len(self.frames) if source.startswith("frame_") else len(self.scenes)
        limit = min(max(1, limit), entity_count, 16384)
        return self.milvus.search(
            collection_name=collection,
            data=np.asarray(queries, dtype=np.float32).tolist(),
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {
                "ef": min(entity_count, 32768, max(64, limit))}},
            limit=limit,
        )

    @staticmethod
    def _sparse_scores(rows: int, hits: list[dict[str, Any]]) -> np.ndarray:
        scores = np.full(rows, -np.inf, dtype=np.float32)
        for hit in hits:
            row = int(hit["id"])
            score = float(hit["distance"])
            if 0 <= row < rows and score > scores[row]:
                scores[row] = score
        return scores

    @staticmethod
    def _candidate_order(scores: np.ndarray, limit: int) -> list[int]:
        rows = np.flatnonzero(np.isfinite(scores))
        if not len(rows):
            return []
        return rows[np.argsort(-scores[rows])[:limit]].astype(int).tolist()

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

    def _exact_ocr_evidence(self, query: str, exact_terms: tuple[str, ...] = ()) -> dict[int, int]:
        """Map scene rows to frames containing queried discriminative numbers."""
        folded = unicodedata.normalize("NFKD", query.casefold()).replace("đ", "d")
        folded = "".join(char for char in folded if not unicodedata.combining(char))
        if not any(cue in folded for cue in ("so ao", "so hieu", "bien so", "ghi so")):
            return {}
        tokens = list(dict.fromkeys([
            *re.findall(r"(?<!\d)\d{2,4}(?!\d)", query),
            *(term for term in exact_terms if re.fullmatch(r"\d{2,4}", term)),
        ]))
        combined: dict[int, int] = {}
        for token in tokens:
            cached = self._exact_ocr_cache.get(token)
            if cached is None:
                cached = {}
                for video_id in self.scenes.video_id.astype(str).unique():
                    path = self.root / "workspace" / "ocr_fullres" / video_id / "regions.parquet"
                    if not path.exists():
                        continue
                    regions = pd.read_parquet(path, columns=["frame_id", "text_raw", "confidence"])
                    matches = regions[regions.text_raw.astype(str).str.fullmatch(
                        re.escape(token), case=False, na=False)]
                    if matches.empty:
                        continue
                    scene_rows = self.scenes[self.scenes.video_id.astype(str).eq(video_id)]
                    for match in matches.sort_values("confidence", ascending=False).itertuples(index=False):
                        frame_id = int(match.frame_id)
                        containing = scene_rows[(scene_rows.start_frame <= frame_id)
                                                & (scene_rows.end_frame_exclusive > frame_id)]
                        if containing.empty:
                            continue
                        row = int(containing.index[0])
                        cached.setdefault(row, frame_id)
                self._exact_ocr_cache[token] = cached
            combined.update(cached)
        return combined

    def _frame_hits_to_scene_scores(self, source: str, vectors: np.ndarray, pooling: str,
                                    candidate_k: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        scene_scores = np.full(len(self.scenes), -np.inf, dtype=np.float32)
        best_frames = np.asarray([
            int(values[0]) if isinstance(values, (list, tuple, np.ndarray)) and len(values) else -1
            for values in self.scenes.representative_frame_ids
        ], dtype=np.int64)
        best_frame_scores = np.full(len(self.scenes), -np.inf, dtype=np.float32)
        scene_candidates: list[list[tuple[float, int]]] = [[] for _ in range(len(self.scenes))]
        frame_limit = min(len(self.frames), 16384, max(100, candidate_k * 8))
        results = self._milvus_search(source, vectors, frame_limit)
        frame_scene_rows = self.frames.scene_row.to_numpy(dtype=np.int64)
        frame_ids = self.frames.frame_id.to_numpy(dtype=np.int64)
        hit_scores: dict[int, float] = {}
        hit_scenes: dict[int, int] = {}
        hit_frames: dict[int, int] = {}
        for hits in results:
            for hit in hits:
                frame_row = int(hit["id"])
                if frame_row < 0 or frame_row >= len(self.frames):
                    raise ValueError(f"Milvus returned invalid frame_row {frame_row}")
                score = float(hit["distance"])
                scene_row = int(frame_scene_rows[frame_row])
                frame_id = int(frame_ids[frame_row])
                if score > hit_scores.get(frame_row, -np.inf):
                    hit_scores[frame_row] = score
                    hit_scenes[frame_row] = scene_row
                    hit_frames[frame_row] = frame_id
        frame_scores, scene_rows = [], []
        for frame_row, score in hit_scores.items():
            scene_row, frame_id = hit_scenes[frame_row], hit_frames[frame_row]
            frame_scores.append(score)
            scene_rows.append(scene_row)
            scene_candidates[scene_row].append((score, frame_id))
            if score > best_frame_scores[scene_row]:
                best_frames[scene_row] = frame_id
                best_frame_scores[scene_row] = score
        if frame_scores:
            pooled = aggregate_scene_scores(np.asarray(frame_scores), np.asarray(scene_rows),
                                            len(self.scenes), pooling)
            present = np.unique(scene_rows)
            scene_scores[present] = pooled[present]
        top_frames = []
        for candidates in scene_candidates:
            selected = []
            for _, frame_id in sorted(candidates, reverse=True):
                if all(abs(frame_id - existing) >= 10 for existing in selected):
                    selected.append(frame_id)
                if len(selected) == 4:
                    break
            top_frames.append(selected)
        return scene_scores, best_frames, top_frames

    def _siglip_scores(self, variants: list[str], pooling: str,
                       candidate_k: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        inputs = self.siglip_processor(text=variants, padding="max_length", truncation=True, max_length=64,
                                       return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vectors = self.siglip.get_text_features(**inputs).float()
            vectors = torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()
        if vectors.shape[1] != self.vector_dimensions["frame_siglip"]:
            raise ValueError("SigLIP query dimension does not match the Milvus frame collection")
        frame_scores, best_frames, top_frames = self._frame_hits_to_scene_scores(
            "frame_siglip", vectors, pooling, candidate_k)
        if "scene_siglip" in self.milvus_collections:
            scene_results = self._milvus_search(
                "scene_siglip", vectors, min(candidate_k, len(self.scenes)))
            scene_scores = np.full(len(self.scenes), -np.inf, dtype=np.float32)
            for hits in scene_results:
                scene_scores = np.maximum(
                    scene_scores, self._sparse_scores(len(self.scenes), hits))
            frame_scores = np.maximum(frame_scores, scene_scores)
        return frame_scores, best_frames, top_frames

    def _bge_scores(self, query: str, candidate_k: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        inputs = self.bge_tokenizer([query], padding=True, truncation=True, max_length=512,
                                    return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vector = self.bge(**inputs).last_hidden_state[:, 0].float()
            vector = torch.nn.functional.normalize(vector, dim=-1).cpu().numpy()[0]
        if vector.shape[0] != self.vector_dimensions["bge_retrieval"]:
            raise ValueError("BGE query dimension does not match the Milvus scene collection")
        primary_hits = self._milvus_search("bge_retrieval", vector[None, :], candidate_k)[0]
        primary = self._sparse_scores(len(self.scenes), primary_hits)
        fields = {}
        for name, source in self.field_text_collections.items():
            hits = self._milvus_search(source, vector[None, :], candidate_k)[0]
            fields[name.removeprefix("bge_")] = self._sparse_scores(len(self.scenes), hits)
        return primary, fields

    def _openclip_scores(self, variants: list[str], pooling: str,
                         candidate_k: int) -> np.ndarray:
        if not self.openclip_available:
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
        if vectors.shape[1] != self.vector_dimensions["frame_openclip"]:
            raise ValueError("OpenCLIP query dimension does not match the Milvus frame collection")
        scores, _, _ = self._frame_hits_to_scene_scores(
            "frame_openclip", vectors, pooling, candidate_k)
        return scores

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
               openclip: bool = False, visual_pooling: str = "max",
               calibrate_openclip: bool = False, reranker_fusion: str = "rank",
               online_gemini_text: bool = False,
               query_transformer: bool = True,
               progress_callback: Any | None = None) -> dict[str, Any]:
        if not query.strip() or top_k <= 0:
            raise ValueError("query must be non-empty and top_k positive")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        request_started = time.perf_counter()
        with self.lock, torch.inference_mode():
            timings = {"lock_wait": time.perf_counter() - request_started}
            stage_started = time.perf_counter()
            task = route_query(query)
            plan = build_query_plan(task.retrieval_query)
            transformer_error = None
            expansion = None
            if query_transformer:
                if not self.query_transformer.available:
                    raise QueryTransformerUnavailable(
                        "Qwen was requested but is unavailable. Install the configured model and runtime, "
                        "then restart the backend.")
                try:
                    plan, expansion = self.query_transformer.enhance(plan)
                except Exception as exc:
                    if isinstance(exc, (QueryTransformerError, QueryTransformerUnavailable)):
                        raise
                    raise QueryTransformerError(f"Qwen query transformation failed: {exc}") from exc
            weights = intent_weights(plan.intent)
            timings["query_plan"] = time.perf_counter() - stage_started
            variants = list(plan.variants)
            text_first_scores = None
            stage_started = time.perf_counter()
            if online_gemini_text and self.gemini_text_available and task.task == "KIS":
                try:
                    if self.gemini_text_services is None:
                        self.gemini_text_services = GeminiTextServices(self.root, self.index)
                    if isinstance(self.gemini_text_services, GeminiTextServices):
                        expansion = self.gemini_text_services.expand(plan.normalized)
                        variants.extend([expansion.english_translation, *expansion.visual_queries,
                                         " ".join(expansion.keywords_en)])
                        query_vector = self.gemini_text_services.embed_query(
                            plan.normalized, self.vector_dimensions["gemini_text"])
                        hits = self._milvus_search("gemini_text", query_vector[None, :], candidate_k)[0]
                        text_first_scores = self._sparse_scores(len(self.scenes), hits)
                except Exception:
                    self.gemini_text_services = False
            timings["gemini_text"] = time.perf_counter() - stage_started
            variants = list(dict.fromkeys(value.strip() for value in variants if value.strip()))
            stage_started = time.perf_counter()
            visual, best_frames, top_frames = self._siglip_scores(
                variants, visual_pooling, candidate_k)
            timings["siglip"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            semantic, semantic_fields = self._bge_scores(plan.normalized, candidate_k)
            timings["bge"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            openclip_scores = (self._openclip_scores(variants, visual_pooling, candidate_k)
                               if openclip else None)
            timings["openclip"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            if "object_spatial_json" in self.scenes:
                object_scores, object_constraints = structured_object_scores(
                    plan.normalized, self.scenes.object_spatial_json.tolist())
            else:
                object_scores, object_constraints = np.zeros(len(self.scenes), dtype=np.float32), []
            exact_ocr_frames = self._exact_ocr_evidence(plan.normalized, plan.exact_terms)
            for row, frame_id in exact_ocr_frames.items():
                best_frames[row] = frame_id
                top_frames[row] = [frame_id, *(value for value in top_frames[row] if value != frame_id)][:4]
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
            if reranker_fusion not in {"rank", "score", "hybrid", "auto"}:
                raise ValueError("reranker_fusion must be rank, score, hybrid or auto")
            resolved_reranker_fusion = ("rank" if task.task == "QA" else "hybrid") \
                if reranker_fusion == "auto" else reranker_fusion
            lexical_query = plan.ocr_query if plan.intent == "ocr" and plan.ocr_query else plan.normalized
            orders = {
                "visual": self._candidate_order(visual, candidate_k),
                "semantic": self._candidate_order(semantic, candidate_k),
                "lexical": self._fts(lexical_query, "scene_fts", candidate_k),
                "fuzzy": self._fts(lexical_query, "fuzzy_fts", candidate_k),
            }
            if exact_ocr_frames:
                orders["exact_ocr"] = list(exact_ocr_frames)
            exact_metadata_rows = []
            if plan.exact_terms:
                patterns = [re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
                            for term in plan.exact_terms]
                for row_id, text in enumerate(self.scenes.retrieval_text.fillna("").astype(str)):
                    if all(pattern.search(text) for pattern in patterns):
                        exact_metadata_rows.append(row_id)
                if exact_metadata_rows:
                    orders["exact_metadata"] = exact_metadata_rows[:candidate_k]
            if openclip_scores is not None:
                orders["openclip"] = self._candidate_order(openclip_scores, candidate_k)
            if object_constraints:
                # Do not inject zero-evidence scenes into the candidate union:
                # the detector legitimately misses some queried objects.
                object_order = np.argsort(-object_scores)
                orders["object"] = [int(row) for row in object_order
                                    if object_scores[row] > 0][:candidate_k]
            if text_first_scores is not None:
                orders["text_first"] = self._candidate_order(text_first_scores, candidate_k)
            fused: dict[int, float] = {}
            for branch, order in orders.items():
                for rank, row in enumerate(order, 1):
                    branch_weight = (5.0 if branch == "exact_ocr" else
                                     3.0 if branch == "exact_metadata" else weights[branch])
                    fused[row] = fused.get(row, 0.0) + branch_weight / (60 + rank)
            rapid = {row: fuzz.WRatio(lexical_query, str(self.scenes.iloc[row].retrieval_text)) for row in fused}
            for rank, row in enumerate(sorted(rapid, key=rapid.get, reverse=True), 1):
                fused[row] += weights["fuzzy"] / (60 + rank)
            coverage_scores: dict[int, float] = {}
            coverage_fields: dict[int, dict[str, float]] = {}
            if fusion_mode == "normalized":
                visual_unit, semantic_unit = robust_unit(visual), robust_unit(semantic)
                openclip_unit = robust_unit(openclip_scores) if openclip_scores is not None else None
                text_first_unit = robust_unit(text_first_scores) if text_first_scores is not None else None
                for row in fused:
                    coverage, fields = field_coverage(plan.normalized, self.scenes.iloc[row], plan.intent)
                    coverage_scores[row], coverage_fields[row] = coverage, fields
                    # RRF remains the robust backbone. Calibrated dense evidence and
                    # field coverage only break ties; they cannot remove a candidate.
                    fused[row] += (0.006 * weights["visual"] * float(visual_unit[row])
                                   + 0.006 * weights["semantic"] * float(semantic_unit[row])
                                   + 0.010 * coverage)
                    if calibrate_openclip and openclip_unit is not None:
                        fused[row] += 0.006 * weights["openclip"] * float(openclip_unit[row])
                    if object_constraints:
                        fused[row] += 0.015 * weights["object"] * float(object_scores[row])
                    if text_first_unit is not None:
                        fused[row] += 0.008 * weights["text_first"] * float(text_first_unit[row])
                    if row in exact_ocr_frames:
                        fused[row] += 0.08
            preliminary = sorted(fused, key=fused.get, reverse=True)[:rerank_k]
            timings["candidate_fusion"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            has_visual_text = any(bool(str(self.scenes.iloc[row].visual_summary).strip()) for row in preliminary)
            use_reranker = bool(preliminary) and (plan.intent != "visual" or has_visual_text)
            if progress_callback is not None:
                preview_order = sorted(fused, key=fused.get, reverse=True)[:min(top_k, len(fused))]
                try:
                    progress_callback({
                        "event": "ranked", "phase": "fusion", "final": not use_reranker,
                        "reranker_pending": use_reranker,
                        "items": [{
                            "rank": rank, "fusion_score": float(fused[row_id]),
                            "video_id": str(self.scenes.iloc[row_id].video_id),
                            "scene_index": int(self.scenes.iloc[row_id].scene_index),
                            "best_frame_id": int(best_frames[row_id]),
                            "top_visual_frame_ids": top_frames[row_id],
                        } for rank, row_id in enumerate(preview_order, 1)],
                    })
                except Exception:
                    pass
            reranker_scores = self._rerank(plan.normalized, preliminary, rerank_batch_size) if use_reranker else {}
            timings["reranker"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            reranker_order = sorted(reranker_scores, key=reranker_scores.get, reverse=True)
            reranker_unit = robust_unit(np.asarray([reranker_scores[row] for row in reranker_order], dtype=np.float32))
            for rank, (row, unit_score) in enumerate(zip(reranker_order, reranker_unit), 1):
                rank_delta = weights["reranker"] / (60 + rank)
                score_delta = 0.020 * weights["reranker"] * float(unit_score)
                if resolved_reranker_fusion == "rank":
                    fused[row] += rank_delta
                elif resolved_reranker_fusion == "score":
                    fused[row] += score_delta
                else:
                    fused[row] += 0.5 * (rank_delta + score_delta)
            profile = accepted_profile(task.task, plan.normalized, plan.intent)
            profile_scores: dict[int, float] = {}
            if profile and fused:
                # Preserve candidate-generation recall. Learned/profile fusion
                # may only reorder the bounded top pool used during validation.
                base_order = sorted(fused, key=fused.get, reverse=True)
                profile_rows = base_order[:150]
                raw_features = {
                    "visual": np.asarray([visual[row] for row in profile_rows], dtype=np.float32),
                    "openclip": np.asarray([(openclip_scores[row] if openclip_scores is not None else 0.0)
                                            for row in profile_rows], dtype=np.float32),
                    "semantic": np.asarray([semantic[row] for row in profile_rows], dtype=np.float32),
                    "lexical": np.asarray([rapid[row] for row in profile_rows], dtype=np.float32),
                    "coverage": np.asarray([coverage_scores.get(row, 0.0) for row in profile_rows], dtype=np.float32),
                    "object": np.asarray([object_scores[row] for row in profile_rows], dtype=np.float32),
                    "reranker": np.asarray([reranker_scores.get(row, 0.0) for row in profile_rows], dtype=np.float32),
                }
                units = {name: robust_unit(values) for name, values in raw_features.items()}
                for position, row in enumerate(profile_rows):
                    profile_scores[row] = sum(profile.weights[name] * float(units[name][position])
                                              for name in profile.weights)
                profile_order = sorted(profile_rows, key=profile_scores.get, reverse=True)
                ranked = (profile_order + base_order[len(profile_rows):])[:min(top_k, len(fused))]
            else:
                ranked = sorted(fused, key=fused.get, reverse=True)[:min(top_k, len(fused))]
            results = []
            for rank, row_id in enumerate(ranked, 1):
                row = self.scenes.iloc[row_id]
                results.append({"rank": rank, "fusion_score": fused[row_id],
                                "visual_score": (float(visual[row_id]) if np.isfinite(visual[row_id]) else None),
                                "semantic_score": (float(semantic[row_id]) if np.isfinite(semantic[row_id]) else None),
                                "rapidfuzz_score": float(rapid[row_id]),
                                "openclip_score": (float(openclip_scores[row_id])
                                                   if openclip_scores is not None and np.isfinite(openclip_scores[row_id]) else None),
                                "object_spatial_score": (float(object_scores[row_id])
                                                         if object_constraints else None),
                                "profile_score": profile_scores.get(row_id),
                                "text_first_score": (float(text_first_scores[row_id])
                                                     if text_first_scores is not None else None),
                                "semantic_field_scores": {name: float(values[row_id])
                                                          for name, values in semantic_fields.items()
                                                          if np.isfinite(values[row_id])},
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
                                "ocr_evidence": (json.loads(str(row.ocr_evidence_json))
                                                 if "ocr_evidence_json" in self.scenes.columns
                                                 and not pd.isna(row.ocr_evidence_json) else []),
                                "ocr_text": ([str(value) for value in row.ocr_text]
                                             if isinstance(row.ocr_text, (list, np.ndarray)) else []),
                                "asr_text": "" if pd.isna(row.asr_text) else str(row.asr_text),
                                "scene_context": str(row.retrieval_text)[:600]})
            submissions = build_ranked_submissions(task.task, results, min(100, top_k)) if task.task != "TRAKE" else []
            timings["result_assembly"] = time.perf_counter() - stage_started
            timings["total"] = time.perf_counter() - request_started
            return {"query": query, "task_plan": task.to_dict(), "query_plan": plan.to_dict(), "weights": weights,
                    "fusion_mode": fusion_mode, "field_aware": field_aware, "openclip_used": openclip,
                    "visual_pooling": visual_pooling,
                    "calibrate_openclip": calibrate_openclip, "reranker_fusion": resolved_reranker_fusion,
                    "object_constraints": object_constraints,
                    "exact_ocr_evidence": [{"scene_row": row, "frame_id": frame}
                                           for row, frame in exact_ocr_frames.items()],
                    "retrieval_profile": profile.kind if profile else None,
                    "query_expansion": expansion.model_dump() if expansion else None,
                    "query_transformer_requested": query_transformer,
                    "query_transformer_available": self.query_transformer.available,
                    "query_transformer_used": plan.transformer_used,
                    "query_transformer_error": transformer_error,
                    "online_gemini_text": online_gemini_text,
                    "timings": {name: round(value, 4) for name, value in timings.items()},
                    "reranker_used": use_reranker, "results": results, "submissions": submissions,
                    "requires_temporal_alignment": task.task == "TRAKE"}

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "device": str(self.device), "scenes": len(self.scenes),
                "frames": len(self.frames), "milvus_uri": str(self.milvus_uri),
                "milvus_collections": sorted(self.milvus_collections),
                "image_dimension": self.vector_dimensions.get("frame_siglip", 0),
                "text_dimension": self.vector_dimensions.get("bge_retrieval", 0),
                "semantic_fields": sorted(name.removeprefix("bge_")
                                           for name in self.field_text_collections),
                "gemini_text_available": self.gemini_text_available,
                "openclip_available": self.openclip_available,
                "query_transformer_available": self.query_transformer.available,
                "query_transformer": self.query_transformer.status(),
                "cuda_memory_allocated": int(torch.cuda.memory_allocated()) if self.device.type == "cuda" else 0}

    def close(self) -> None:
        self.milvus.close()
        self.query_transformer.close()
