from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel


class ExpandedQuery(BaseModel):
    english_translation: str
    visual_queries: list[str]
    keywords_vi: list[str]
    keywords_en: list[str]


class GeminiTextServices:
    def __init__(self, root: Path, index: Path,
                 embedding_model: str = "gemini-embedding-001",
                 expansion_model: str = "gemini-3.5-flash-lite") -> None:
        self.root, self.index = root.resolve(), index.resolve()
        self.embedding_model, self.expansion_model = embedding_model, expansion_model
        load_dotenv(self.root / ".env")
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is not configured")
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.database = self.index / "gemini_text_cache.sqlite3"
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, kind TEXT, value TEXT)")

    def _get(self, key: str) -> str | None:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None

    def _put(self, key: str, kind: str, value: str) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?)", (key, kind, value))

    def expand(self, query: str) -> ExpandedQuery:
        from google.genai import types
        key = hashlib.sha256(f"expand:{self.expansion_model}:{query}".encode("utf-8")).hexdigest()
        cached = self._get(key)
        if cached:
            return ExpandedQuery.model_validate_json(cached)
        prompt = f"""Expand this Vietnamese AIC video retrieval query without inventing details: {query}
Return a faithful English translation, up to 3 concise visual paraphrases, and short Vietnamese/English keywords.
Preserve exact counts, left/right/top/bottom positions, names, OCR strings and event order."""
        response = self.client.models.generate_content(
            model=self.expansion_model, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=ExpandedQuery, temperature=0,
                                               max_output_tokens=512,
                                               thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")))
        value = ExpandedQuery.model_validate_json(response.text)
        self._put(key, "expansion", value.model_dump_json())
        return value

    def embed_query(self, query: str, dimension: int) -> np.ndarray:
        from google.genai import types
        key = hashlib.sha256(
            f"embed:{self.embedding_model}:{dimension}:{query}".encode("utf-8")).hexdigest()
        cached = self._get(key)
        if cached:
            vector = np.asarray(json.loads(cached), dtype=np.float32)
        else:
            response = self.client.models.embed_content(
                model=self.embedding_model, contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY",
                                                output_dimensionality=dimension))
            vector = np.asarray(response.embeddings[0].values, dtype=np.float32)
            self._put(key, "embedding", json.dumps(vector.tolist()))
        return vector / max(float(np.linalg.norm(vector)), 1e-12)
