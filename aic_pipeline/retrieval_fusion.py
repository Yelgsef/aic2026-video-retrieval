from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np


STOPWORDS = {
    "và", "với", "có", "một", "những", "nhiều", "của", "trong", "trên", "tại", "là",
    "đang", "được", "cho", "cảnh", "quay", "hình", "ảnh", "thấy", "về", "khi", "này",
}


def fold_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    return " ".join("".join(c if c.isalnum() else " " for c in text
                             if not unicodedata.combining(c)).split())


def content_terms(query: str) -> list[str]:
    return list(dict.fromkeys(term for term in fold_text(query).split()
                              if len(term) >= 2 and term not in STOPWORDS))


def robust_unit(scores: np.ndarray) -> np.ndarray:
    """Map a score vector to a stable [0, 1] range without dataset-specific constants."""
    values = np.asarray(scores, dtype=np.float32)
    finite = np.isfinite(values)
    output = np.zeros_like(values)
    if not finite.any():
        return output
    low, high = np.quantile(values[finite], [0.50, 0.995])
    if high <= low + 1e-8:
        return output
    output[finite] = np.clip((values[finite] - low) / (high - low), 0.0, 1.0)
    return output


def field_coverage(query: str, row: Any, intent: str) -> tuple[float, dict[str, float]]:
    terms = content_terms(query)
    if not terms:
        return 0.0, {}
    raw_ocr = getattr(row, "ocr_text", [])
    if raw_ocr is None or (isinstance(raw_ocr, float) and np.isnan(raw_ocr)):
        ocr_values: list[str] = []
    elif isinstance(raw_ocr, str):
        ocr_values = [raw_ocr]
    else:
        ocr_values = [str(value) for value in raw_ocr]
    fields = {
        "summary": fold_text(getattr(row, "visual_summary", "")),
        "asr": fold_text(f"{getattr(row, 'asr_text', '')} {getattr(row, 'asr_context', '')}"),
        "ocr": fold_text(" ".join(ocr_values)),
        "entities": fold_text(getattr(row, "entities_text", "")),
    }
    scores = {name: sum(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None
                        for term in terms) / len(terms) for name, text in fields.items()}
    weights = {"summary": 1.0, "asr": 0.7, "ocr": 0.7, "entities": 0.8}
    if intent == "ocr":
        weights.update(ocr=1.5, entities=1.0, asr=0.4)
    elif intent == "asr":
        weights.update(asr=1.5, summary=0.8, ocr=0.3)
    numerator = sum(weights[name] * score for name, score in scores.items())
    return numerator / sum(weights.values()), scores
