from __future__ import annotations

import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field

from .object_grounding import GroundingDinoCropper
from .qa_reasoning import answers_equivalent, classify_question, temporal_frame_ids
from .router_chat import RouterChatClient


class QASelection(BaseModel):
    candidate_label: str = Field(description="Exactly one candidate label, for example C1")
    answer: str = Field(description="Short answer in Vietnamese or English")
    evidence_vi: str
    confidence: float = Field(ge=0, le=1)


class QASceneShortlist(BaseModel):
    candidate_labels: list[str] = Field(
        min_length=1, max_length=3,
        description="One to three candidate labels ordered by visual scene match",
    )
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
        (r",?\s*đồng hồ(?:\s+thời gian)?(?:\s+ở\s+góc\s+màn\s+hình)?\s+"
         r"(?:đang\s+)?(?:chỉ|hiển\s+thị)?\s*(?:mấy\s+giờ|bao\s+nhiêu).*$", ""),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return " ".join(text.split())


def local_exact_ocr_answer(root: Path, question: str,
                           result: dict[str, Any]) -> dict[str, Any] | None:
    """Answer exact clock/time questions from OCR on the retrieved frame."""
    folded = unicodedata.normalize("NFKD", question.casefold()).replace("đ", "d")
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    if not any(cue in folded for cue in ("may gio", "dong ho", "thoi gian")):
        return None
    video_id, frame_id = str(result["video_id"]), int(result["best_frame_id"])
    path = root / "workspace" / "ocr_fullres" / video_id / "regions.parquet"
    if not path.exists():
        return None
    regions = pd.read_parquet(path)
    frame = regions[regions.frame_id.astype(int).eq(frame_id)].copy()
    if frame.empty:
        return None
    frame["clock_text"] = frame.text_raw.astype(str).str.extract(
        r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$", expand=False)
    matches = frame[frame.clock_text.notna()].sort_values("confidence", ascending=False)
    if matches.empty:
        return None
    row = matches.iloc[0]
    answer = str(row.clock_text)
    return {"task": "QA", "video_id": video_id, "frame_id": frame_id,
            "scene_index": int(result["scene_index"]), "answer": answer,
            "confidence": float(row.confidence),
            "evidence_vi": f"PP-OCR đọc đồng hồ trên frame {frame_id}: {answer}",
            "candidate_label": "LOCAL_OCR", "model": "PP-OCRv6",
            "candidate_count": 1, "refined_answer": False,
            "object_grounding_used": False, "object_crop_count": 0,
            "qa_type": "ocr_time", "temporal_evidence_used": False,
            "first_answer": answer, "answers_agree": True,
            "adjudicated": False, "refinement_requested": False,
            "timings": {"total": 0.0}}


class GeminiQAEngine:
    def __init__(self, root: Path, model: str | None = None,
                 request_timeout_ms: int = 20_000) -> None:
        self.root = root.resolve()
        load_dotenv(self.root / ".env")
        self.client = RouterChatClient(self.root, request_timeout_ms / 1000)
        self.model = model or self.client.model
        self.manifest = pd.read_parquet(self.root / "workspace" / "manifests" / "videos.parquet").set_index("video_id")
        self._grounder: GroundingDinoCropper | bool | None = None
        self._frame_cache: dict[tuple[str, int], Image.Image] = {}

    def _get_grounder(self) -> GroundingDinoCropper | None:
        if self._grounder is None:
            model_dir = self.root / "models" / "grounding-dino-tiny"
            if not (model_dir / "config.json").exists():
                self._grounder = False
            else:
                try:
                    self._grounder = GroundingDinoCropper(model_dir)
                except Exception:
                    self._grounder = False
        return self._grounder if isinstance(self._grounder, GroundingDinoCropper) else None

    def _decode(self, video_id: str, frame_id: int) -> Image.Image:
        key = (video_id, frame_id)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached.copy()
        path = str(self.manifest.loc[video_id].video_path)
        capture = cv2.VideoCapture(path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Cannot decode {video_id} frame {frame_id}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(self._frame_cache) >= 256:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[key] = image
        return image.copy()

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
            source = candidate.get("_image")
            if not isinstance(source, Image.Image):
                source = self._decode(str(candidate["video_id"]), int(candidate["best_frame_id"]))
            image = ImageOps.fit(source,
                                 (tile_w, tile_h), method=Image.Resampling.LANCZOS)
            x, y = index % cols * tile_w, index // cols * (tile_h + label_h)
            sheet.paste(image, (x, y))
            crop_label = f" | crop:{candidate.get('_object_label')}" if candidate.get("_object_crop") else ""
            draw.text((x + 6, y + tile_h + 8),
                      f"{label} | {candidate['video_id']} | frame {candidate['best_frame_id']}{crop_label}", fill="black")
        return sheet, mapping

    @staticmethod
    def _spatial_text_crops(image: Image.Image, question: str) -> list[tuple[str, Image.Image]]:
        """Create QA-only crops for small text referenced by spatial language."""
        folded = unicodedata.normalize("NFKD", question.casefold()).replace("đ", "d")
        folded = "".join(char for char in folded if not unicodedata.combining(char))
        width, height = image.size
        crops: list[tuple[str, Image.Image]] = []
        if any(cue in folded for cue in ("ben phai", "phia phai", "goc phai")):
            crops.append(("right-side-text", image.crop((int(width * 0.38), 0, width, height))))
        if any(cue in folded for cue in ("ben trai", "phia trai", "goc trai")):
            crops.append(("left-side-text", image.crop((0, 0, int(width * 0.62), height))))
        if any(cue in folded for cue in ("phia tren", "ben tren", "o tren")):
            crops.append(("upper-text", image.crop((0, 0, width, int(height * 0.62)))))
        if any(cue in folded for cue in ("phia duoi", "ben duoi", "o duoi")):
            crops.append(("lower-text", image.crop((0, int(height * 0.38), width, height))))
        return crops

    def answer(self, question: str, retrieval_results: list[dict[str, Any]], maximum_candidates: int = 24) -> dict[str, Any]:
        answer_started = time.perf_counter()
        timings: dict[str, float] = {}
        reasoning = classify_question(question)
        candidates = []
        # First pass normally maximizes scene recall with one strong frame per
        # scene. Exact-text questions need a different trade-off: a scene can
        # contain a book/page/title for only a few frames, so inspect several
        # temporally diverse views from each of the six strongest scenes. This
        # keeps the same 24-image budget while avoiding a single-frame OCR miss.
        views_per_scene = 4 if reasoning.needs_exact_text else 1
        candidate_budget = max(maximum_candidates, 24) if reasoning.needs_exact_text else maximum_candidates
        scene_limit = max(1, candidate_budget // views_per_scene)
        ranked_results = list(retrieval_results)
        if reasoning.kind == "named_entity":
            # Answer-slot queries such as "cây nào" do not contain the entity
            # name, so pure similarity can rank the correct OCR scene below
            # descriptive but answer-less scenes. Promote, but do not invent,
            # candidates carrying an explicit typed name in OCR.
            typed_name = re.compile(r"\b(?:cây|rau)\s+[\wÀ-ỹ-]+", re.IGNORECASE)
            evidence_results = [item for item in ranked_results
                                if any(typed_name.search(str(value))
                                       for value in item.get("ocr_text", []))]
            evidence_ids = {id(item) for item in evidence_results}
            ranked_results = [*evidence_results,
                              *(item for item in ranked_results if id(item) not in evidence_ids)]
        if reasoning.needs_exact_text and ranked_results:
            # Book pages, signs and captions frequently straddle a shot/scene
            # boundary. Put adjacent scenes from the top video's event directly
            # after rank 1 so their high-resolution views are not crowded out.
            anchor = ranked_results[0]
            adjacent = [item for item in ranked_results[1:]
                        if item.get("video_id") == anchor.get("video_id")
                        and abs(int(item.get("scene_index", -9999))
                                - int(anchor.get("scene_index", 9999))) == 1]
            adjacent_ids = {id(item) for item in adjacent}
            ranked_results = [anchor, *adjacent,
                              *(item for item in ranked_results[1:] if id(item) not in adjacent_ids)]
        for scene_rank, result in enumerate(ranked_results[:candidate_budget], start=1):
            if scene_rank > scene_limit:
                break
            visual_ids = [int(value) for value in result.get("top_visual_frame_ids", [])]
            representative_ids = [int(value) for value in result.get("representative_frame_ids", [])]
            typed_ocr_ids = []
            if reasoning.kind == "named_entity":
                for evidence in result.get("ocr_evidence", []):
                    if typed_name.search(str(evidence.get("text", ""))):
                        typed_ocr_ids.append(int(evidence["frame_id"]))
            if reasoning.needs_exact_text and representative_ids:
                # Prefer the visual best, then evenly cover the scene. With six
                # representatives this selects approximately start/middle/end.
                slots = views_per_scene - 1
                if slots == 1:
                    sampled_representatives = [representative_ids[len(representative_ids) // 2]]
                else:
                    sampled_representatives = [
                        representative_ids[round(index * (len(representative_ids) - 1) / (slots - 1))]
                        for index in range(slots)
                    ]
                frame_ids = typed_ocr_ids + visual_ids[:1] + sampled_representatives
            else:
                frame_ids = visual_ids[:1]
            if not frame_ids:
                frame_ids = [int(result["best_frame_id"])]
            for frame_id in dict.fromkeys(frame_ids):
                candidate = dict(result)
                candidate["best_frame_id"] = frame_id
                candidate["retrieval_scene_rank"] = scene_rank
                candidates.append(candidate)
                if len(candidates) >= candidate_budget:
                    break
            if len(candidates) >= candidate_budget:
                break
        if not candidates:
            raise ValueError("No retrieval candidates")
        stage_started = time.perf_counter()
        sheet, mapping = self._sheet(candidates)
        timings["candidate_decode_sheet"] = time.perf_counter() - stage_started
        prompt = f"""Answer this AIC visual question using only the candidate frames.
Question: {question}
First select the single frame that best matches the described event, then answer the question from visible evidence.
Question type: {reasoning.kind}. Specialized instruction: {reasoning.instruction}
Rules:
- candidate_label must be one of {', '.join(mapping)}.
- Retrieval rank is a useful prior: prefer evidence from a better-ranked scene when visual evidence is comparable.
- Do not use outside knowledge or infer hidden information.
- Keep answer short and in Vietnamese or English.
- If the answer is unclear, choose the best candidate but set confidence below 0.5 and explain the uncertainty.
"""
        response, used_model, last_error = None, self.model, None
        stage_started = time.perf_counter()
        for model in (self.model,):
            for attempt in range(2):
                try:
                    first_prompt = prompt
                    first_schema: type[BaseModel] = QASelection
                    if reasoning.needs_exact_text:
                        first_prompt = f"""Shortlist visual scenes for an exact-text AIC question.
Question: {question}
Use every visual constraint in the question: people, clothing, objects, colors, layout, left/right position and background.
This pass is ONLY scene grounding. Do not answer, translate or infer the requested text.
Return 1-3 candidate labels ordered by how completely the frame satisfies the whole scene description.
The labels must come from: {', '.join(mapping)}.
Semantic topic similarity alone is not sufficient. A frame about the same topic but with the wrong layout is a hard negative.
"""
                        first_schema = QASceneShortlist
                    response = self.client.json_response(
                        first_prompt, first_schema, sheet, model=model,
                        temperature=0.1, max_tokens=1024)
                    used_model = model
                    break
                except Exception as exc:
                    last_error = exc
                    if not any(code in str(exc) for code in (
                            "429", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")):
                        raise
                    time.sleep(3 * (attempt + 1))
            if response is not None:
                break
        if response is None:
            raise RuntimeError(f"Gemini QA failed after retries: {last_error}")
        timings["gemini_first_pass"] = time.perf_counter() - stage_started
        shortlist_labels: list[str] = []
        if reasoning.needs_exact_text:
            shortlist = QASceneShortlist.model_validate_json(response.text)
            shortlist_labels = [value.strip().upper() for value in shortlist.candidate_labels]
            shortlist_labels = list(dict.fromkeys(value for value in shortlist_labels if value in mapping))
            if not shortlist_labels:
                raise ValueError("Gemini returned no valid QA scene shortlist labels")
            label = shortlist_labels[0]
            candidate = mapping[label]
            selection = QASelection(candidate_label=label, answer="", evidence_vi=shortlist.evidence_vi,
                                    confidence=shortlist.confidence)
        else:
            selection = QASelection.model_validate_json(response.text)
            label = selection.candidate_label.strip().upper()
            if label not in mapping:
                raise ValueError(f"Gemini returned invalid candidate label: {label}")
            candidate = mapping[label]
        initial_selection = selection
        refined_answer = False
        # Second pass is expensive. Use it only for temporal/object/OCR-like
        # questions or when Gemini itself reports weak confidence.
        # Exact-text first pass already receives up to four views per scene.
        # Avoid a second API call when that richer pass is highly confident;
        # retain refinement for uncertain OCR and all temporal questions.
        needs_refinement = (reasoning.needs_temporal or reasoning.needs_exact_text
                            or selection.confidence < 0.65)
        # Second pass: inspect several full-size views from only the selected
        # scene. This preserves broad scene recall while making OCR and small
        # objects much easier than on a 24-tile contact sheet.
        if not needs_refinement:
            detail_ids = []
        elif reasoning.needs_temporal:
            detail_ids = temporal_frame_ids(
                int(candidate["start_frame"]), int(candidate["end_frame_exclusive"]),
                int(candidate["best_frame_id"]), maximum=5)
        else:
            detail_ids = []
        detail_candidates: list[dict[str, Any]] = []
        stage_started = time.perf_counter()
        detail_sources = [candidate]
        if reasoning.needs_exact_text:
            detail_sources = [mapping[value] for value in shortlist_labels]
        for source_candidate in detail_sources:
            source_ids = ([int(source_candidate["best_frame_id"])]
                          + [int(value) for value in source_candidate.get("top_visual_frame_ids", [])]
                          + [int(value) for value in source_candidate.get("representative_frame_ids", [])])
            source_limit = 4 if reasoning.needs_exact_text else (5 if reasoning.needs_temporal else 4)
            selected_source_ids: list[int] = []
            for frame_id in source_ids:
                if frame_id not in selected_source_ids:
                    selected_source_ids.append(frame_id)
                if len(selected_source_ids) >= source_limit:
                    break
            for frame_id in selected_source_ids:
                item = dict(source_candidate)
                item["best_frame_id"] = frame_id
                item["_image"] = self._decode(str(item["video_id"]), frame_id)
                detail_ids.append(frame_id)
                # Keep the full frame for scene grounding and add a directional
                # crop for exact transcription of small signs/pages.
                if reasoning.needs_exact_text:
                    for crop_label, crop in self._spatial_text_crops(item["_image"], question):
                        crop_item = dict(item)
                        crop_item["_image"] = crop.resize(
                            (crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
                        crop_item["_object_crop"] = True
                        crop_item["_object_label"] = crop_label
                        detail_candidates.append(crop_item)
                detail_candidates.append(item)
        if not reasoning.needs_exact_text:
            detail_candidates = []
            for frame_id in detail_ids:
                item = dict(candidate)
                item["best_frame_id"] = frame_id
                item["_image"] = self._decode(str(item["video_id"]), frame_id)
                detail_candidates.append(item)
        timings["detail_decode"] = time.perf_counter() - stage_started
        object_crop_count = 0
        # Directional text crops are deterministic and much cheaper than
        # loading GroundingDINO. Reserve object grounding for non-text QA.
        grounder = self._get_grounder() if reasoning.needs_object_crop and not reasoning.needs_exact_text else None
        if grounder and detail_candidates and reasoning.needs_object_crop:
            try:
                grounded = grounder.crop([item["_image"] for item in detail_candidates], question, maximum=3)
                for detection in grounded:
                    base = dict(detail_candidates[int(detection["image_index"])])
                    base["_image"] = detection["image"]
                    base["_object_crop"] = True
                    base["_object_label"] = detection["label"]
                    base["_object_score"] = detection["score"]
                    detail_candidates.append(base)
                    object_crop_count += 1
                    # Preserve subject-object relations with a second, wider crop.
                    source = detail_candidates[int(detection["image_index"])]["_image"]
                    x1, y1, x2, y2 = detection["box"]
                    width, height = x2 - x1, y2 - y1
                    context = source.crop((max(0, int(x1 - width)), max(0, int(y1 - height)),
                                           min(source.width, int(x2 + width)),
                                           min(source.height, int(y2 + height))))
                    context_item = dict(base)
                    context_item["_image"] = context
                    context_item["_object_label"] = f"{detection['label']}:context"
                    detail_candidates.append(context_item)
            except Exception:
                object_crop_count = 0
        if detail_candidates:
            detail_sheet, detail_mapping = self._sheet(detail_candidates)
            detail_prompt = f"""Answer the visual question from these high-resolution candidate views.
Question: {question}
Question type: {reasoning.kind}. Specialized instruction: {reasoning.instruction}
Select the frame that satisfies the complete scene description, not merely the same topic.
Read exact visible text carefully. Preserve the original language, accents, spelling and word order.
Do not translate, paraphrase, summarize or invent text from the topic. If the requested text is not legible, answer UNCERTAIN with confidence below 0.5.
candidate_label must be one of {', '.join(detail_mapping)}. Keep the answer short.
"""
            stage_started = time.perf_counter()
            try:
                detail_response = self.client.json_response(
                    detail_prompt, QASelection, detail_sheet, model=used_model,
                    temperature=0.0, max_tokens=1024)
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
            timings["gemini_refinement"] = time.perf_counter() - stage_started
        adjudicated = False
        answers_agree = (answers_equivalent(initial_selection.answer, selection.answer)
                         if initial_selection.answer else True)
        if refined_answer and initial_selection.answer and not answers_agree and detail_candidates:
            # A third Gemini adjudication made tail latency unbounded. Resolve
            # locally: retain the answer with higher self-reported confidence.
            # Refinement is a zoomed verification pass, not permission to
            # replace an equally confident, broader-context OCR reading. This
            # prevents a nearby red chapter marker from displacing the actual
            # title selected in the event-level pass.
            if initial_selection.confidence >= selection.confidence:
                selection = initial_selection
                label = initial_selection.candidate_label.strip().upper()
                candidate = mapping[label]
        timings["total"] = time.perf_counter() - answer_started
        return {"task": "QA", "video_id": str(candidate["video_id"]),
                "frame_id": int(candidate["best_frame_id"]), "scene_index": int(candidate["scene_index"]),
                "answer": selection.answer, "confidence": selection.confidence,
                "evidence_vi": selection.evidence_vi, "candidate_label": label,
                "model": used_model, "candidate_count": len(candidates),
                "refined_answer": refined_answer, "object_grounding_used": object_crop_count > 0,
                "object_crop_count": object_crop_count, "qa_type": reasoning.kind,
                "temporal_evidence_used": reasoning.needs_temporal,
                "first_answer": initial_selection.answer, "answers_agree": answers_agree,
                "adjudicated": adjudicated, "refinement_requested": needs_refinement,
                "timings": {name: round(value, 4) for name, value in timings.items()}}
