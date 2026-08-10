from __future__ import annotations

import os
import re
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


def qa_grounding_query(question: str) -> str:
    """Remove answer-slot wording while preserving the visual scene description."""
    text = question.strip().rstrip("?.! ")
    replacements = (
        (r"\bmàu gì\b", ""), (r"\bvật gì\b", "đồ vật"),
        (r"\bloài động vật nào\b", "động vật"), (r"\bcon vật nào\b", "động vật"),
        (r"\bhiện tượng gì\b", "hiện tượng"), (r"\bhành động gì\b", "hành động"),
        (r"\btheo hướng nào\b", ""), (r"\blà gì\b", ""),
        (r"\bghi chữ và số gì\b", ""),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return " ".join(text.split())


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
        candidates = []
        # Cover more scenes while showing two temporal views per scene. A single
        # pooled best frame often misses the small object asked about in QA.
        for scene_rank, result in enumerate(retrieval_results[: max(1, maximum_candidates // 2)], start=1):
            frame_ids = [int(value) for value in result.get("top_visual_frame_ids", [])[:2]]
            if not frame_ids:
                frame_ids = [int(result["best_frame_id"])]
            for frame_id in frame_ids:
                candidate = dict(result)
                candidate["best_frame_id"] = frame_id
                candidate["retrieval_scene_rank"] = scene_rank
                candidates.append(candidate)
                if len(candidates) >= maximum_candidates:
                    break
            if len(candidates) >= maximum_candidates:
                break
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
        for model in (self.model,):
            for attempt in range(1):
                try:
                    response = self.client.models.generate_content(
                        model=model, contents=[prompt, sheet],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json", response_schema=QASelection,
                            temperature=0.1, max_output_tokens=512,
                            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")))
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
        refined_answer = False
        # Second pass: inspect several full-size views from only the selected
        # scene. This preserves broad scene recall while making OCR and small
        # objects much easier than on a 24-tile contact sheet.
        detail_ids = []
        for frame_id in (candidate.get("top_visual_frame_ids", [])
                         + candidate.get("representative_frame_ids", [])):
            frame_id = int(frame_id)
            if frame_id not in detail_ids:
                detail_ids.append(frame_id)
            if len(detail_ids) == 4:
                break
        detail_candidates = []
        for frame_id in detail_ids:
            item = dict(candidate)
            item["best_frame_id"] = frame_id
            detail_candidates.append(item)
        if detail_candidates:
            detail_sheet, detail_mapping = self._sheet(detail_candidates)
            detail_prompt = f"""Answer the visual question from these high-resolution views of one scene.
Question: {question}
Select the clearest evidence frame. Read exact visible text carefully. Do not infer hidden facts.
candidate_label must be one of {', '.join(detail_mapping)}. Keep the answer short.
"""
            try:
                detail_response = self.client.models.generate_content(
                    model=used_model, contents=[detail_prompt, detail_sheet],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_schema=QASelection,
                        temperature=0.0, max_output_tokens=512,
                        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")))
                refined = QASelection.model_validate_json(detail_response.text)
                refined_label = refined.candidate_label.strip().upper()
                if refined_label in detail_mapping:
                    selection = refined
                    label = refined_label
                    candidate = detail_mapping[refined_label]
                    refined_answer = True
            except Exception:
                # The first pass remains a valid bounded-latency fallback.
                pass
        return {"task": "QA", "video_id": str(candidate["video_id"]),
                "frame_id": int(candidate["best_frame_id"]), "scene_index": int(candidate["scene_index"]),
                "answer": selection.answer, "confidence": selection.confidence,
                "evidence_vi": selection.evidence_vi, "candidate_label": label,
                "model": used_model, "candidate_count": len(candidates),
                "refined_answer": refined_answer}
