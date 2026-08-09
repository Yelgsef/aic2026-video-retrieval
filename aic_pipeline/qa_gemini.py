from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field


class QASelection(BaseModel):
    candidate_label: str = Field(description="Exactly one candidate label, for example C1")
    answer: str = Field(description="Short answer in Vietnamese or English")
    evidence_vi: str
    confidence: float = Field(ge=0, le=1)


class GeminiQAEngine:
    def __init__(self, root: Path, model: str = "gemini-3.6-flash",
                 request_timeout_ms: int = 45_000) -> None:
        self.root = root.resolve()
        self.model = model
        load_dotenv(self.root / ".env")
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        # Bound external latency so one API outage cannot hang the contest service.
        self.client = genai.Client(api_key=key,
                                   http_options=types.HttpOptions(timeout=request_timeout_ms))
        self.manifest = pd.read_parquet(self.root / "workspace" / "manifests" / "videos.parquet").set_index("video_id")

    def _decode(self, video_id: str, frame_id: int) -> Image.Image:
        path = str(self.manifest.loc[video_id].video_path)
        capture = cv2.VideoCapture(path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Cannot decode {video_id} frame {frame_id}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _sheet(self, candidates: list[dict[str, Any]]) -> tuple[Image.Image, dict[str, dict[str, Any]]]:
        cols = 4 if len(candidates) > 8 else 2
        tile_w, tile_h, label_h = ((336, 189, 30) if cols == 4 else (448, 252, 34))
        rows = (len(candidates) + cols - 1) // cols
        sheet = Image.new("RGB", (tile_w * cols, (tile_h + label_h) * rows), "white")
        draw = ImageDraw.Draw(sheet)
        mapping = {}
        for index, candidate in enumerate(candidates):
            label = f"C{index + 1}"
            mapping[label] = candidate
            image = ImageOps.fit(self._decode(str(candidate["video_id"]), int(candidate["best_frame_id"])),
                                 (tile_w, tile_h), method=Image.Resampling.LANCZOS)
            x, y = index % cols * tile_w, index // cols * (tile_h + label_h)
            sheet.paste(image, (x, y))
            draw.text((x + 6, y + tile_h + 8),
                      f"{label} | {candidate['video_id']} | frame {candidate['best_frame_id']}", fill="black")
        return sheet, mapping

    def answer(self, question: str, retrieval_results: list[dict[str, Any]], maximum_candidates: int = 24) -> dict[str, Any]:
        candidates = retrieval_results[:maximum_candidates]
        if not candidates:
            raise ValueError("No retrieval candidates")
        sheet, mapping = self._sheet(candidates)
        prompt = f"""Answer this AIC visual question using only the candidate frames.
Question: {question}
First select the single frame that best matches the described event, then answer the question from visible evidence.
Rules:
- candidate_label must be one of {', '.join(mapping)}.
- Do not use outside knowledge or infer hidden information.
- Keep answer short and in Vietnamese or English.
- If the answer is unclear, choose the best candidate but set confidence below 0.5 and explain the uncertainty.
"""
        response, used_model, last_error = None, self.model, None
        for model in dict.fromkeys((self.model, "gemini-2.5-flash")):
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model, contents=[prompt, sheet],
                        config=types.GenerateContentConfig(response_mime_type="application/json",
                                                           response_schema=QASelection, temperature=0.1))
                    used_model = model
                    break
                except Exception as exc:
                    last_error = exc
                    if not any(code in str(exc) for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                        raise
                    time.sleep(3 * (attempt + 1))
            if response is not None:
                break
        if response is None:
            raise RuntimeError(f"Gemini QA failed after retries: {last_error}")
        selection = QASelection.model_validate_json(response.text)
        label = selection.candidate_label.strip().upper()
        if label not in mapping:
            raise ValueError(f"Gemini returned invalid candidate label: {label}")
        candidate = mapping[label]
        return {"task": "QA", "video_id": str(candidate["video_id"]),
                "frame_id": int(candidate["best_frame_id"]), "scene_index": int(candidate["scene_index"]),
                "answer": selection.answer, "confidence": selection.confidence,
                "evidence_vi": selection.evidence_vi, "candidate_label": label,
                "model": used_model, "candidate_count": len(candidates)}
