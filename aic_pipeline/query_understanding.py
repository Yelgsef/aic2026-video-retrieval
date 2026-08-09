from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


OCR_MARKERS = ("chữ", "dòng chữ", "biển báo", "biển cảnh báo", "logo", "văn bản", "màn hình", "tiêu đề", "ghi")
ASR_MARKERS = ("nói rằng", "đề cập", "lời nói", "nghe thấy", "cho biết rằng", "thông báo rằng")
VISUAL_MARKERS = ("cảnh quay", "cận cảnh", "toàn cảnh", "từ trên cao", "mặc áo", "đứng", "ngồi", "màu")
ENGLISH_GLOSSES = {
    "người đàn ông": "man", "người phụ nữ": "woman", "người dẫn chương trình": "news presenter",
    "trẻ em": "children", "xe máy": "motorbike", "ô tô": "car", "máy bay": "airplane",
    "tàu": "ship", "cần cẩu": "crane", "container": "shipping container", "cảng biển": "seaport",
    "sạt lở": "landslide erosion", "sụt lún": "land subsidence", "lũ": "flood",
    "cháy rừng": "wildfire", "tai nạn": "accident", "phỏng vấn": "interview",
    "đường phố": "street", "bờ biển": "coast", "bờ kênh": "canal bank", "sông": "river",
    "nhà dân": "residential houses", "từ trên cao": "aerial view", "phòng họp": "meeting room",
    "studio": "television studio", "biển cảnh báo": "warning sign", "tạm dừng lưu thông": "road closed",
    "màn hình": "screen", "biển báo": "sign", "dòng chữ": "text", "màu đỏ": "red",
    "màu xanh": "blue", "đứng": "standing", "ngồi": "sitting", "nói": "speaking",
}


@dataclass(frozen=True)
class QueryPlan:
    original: str
    normalized: str
    intent: str
    entities: tuple[str, ...]
    variants: tuple[str, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["entities"] = list(self.entities)
        value["variants"] = list(self.variants)
        return value


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def build_query_plan(query: str) -> QueryPlan:
    normalized = normalize_text(query)
    folded = normalized.casefold()
    intent = "ocr" if any(value in folded for value in OCR_MARKERS) else (
        "visual" if any(value in folded for value in VISUAL_MARKERS) else (
            "asr" if any(value in folded for value in ASR_MARKERS) else "visual"))
    entities = tuple(key for key in ENGLISH_GLOSSES if key in folded)
    english = normalized
    for phrase in sorted(entities, key=len, reverse=True):
        english = re.sub(re.escape(phrase), ENGLISH_GLOSSES[phrase], english, flags=re.IGNORECASE)
    variants = [normalized]
    if english.casefold() != normalized.casefold():
        variants.append(english)
    if entities:
        variants.append(", ".join(ENGLISH_GLOSSES[value] for value in entities))
    return QueryPlan(query, normalized, intent, entities, tuple(dict.fromkeys(variants)))
