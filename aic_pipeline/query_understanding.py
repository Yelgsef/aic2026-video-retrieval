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
    # AIC/VBS-specific visual and exact-text concepts. Longer phrases are
    # replaced first, so e.g. "vận động viên xe đạp" is not split at "xe đạp".
    "vận động viên xe đạp": "competitive cyclist", "người đi xe đạp": "cyclist",
    "xe đạp": "bicycle", "cuộc đua xe đạp": "bicycle race",
    "số áo": "bib number", "số hiệu": "identification number",
    "biển số xe": "license plate", "biển số": "license plate",
    "bảng điểm": "scoreboard", "đồng hồ thời gian": "race clock",
    "đồng hồ": "clock", "phụ đề": "subtitle", "tin chạy": "news ticker",
    "góc trên bên trái": "top left corner", "góc trên bên phải": "top right corner",
    "góc dưới bên trái": "bottom left corner", "góc dưới bên phải": "bottom right corner",
    "ở bên trái": "on the left", "ở bên phải": "on the right",
    "phía trước": "in front of", "phía sau": "behind", "bên cạnh": "next to",
    "đang cầm": "holding", "đang mặc": "wearing", "đội mũ": "wearing a hat",
    "micro": "microphone", "điện thoại": "mobile phone", "máy tính": "computer",
}

OBJECT_GLOSSES = {
    "người": "person", "vận động viên": "person", "xe đạp": "bicycle",
    "xe máy": "motorcycle", "ô tô": "car", "xe buýt": "bus", "tàu": "boat",
    "máy bay": "airplane", "điện thoại": "cell phone", "micro": "microphone",
    "máy tính": "laptop", "chó": "dog", "mèo": "cat",
}


@dataclass(frozen=True)
class QueryPlan:
    original: str
    normalized: str
    intent: str
    visual_query: str
    ocr_query: str
    asr_query: str
    entities: tuple[str, ...]
    variants: tuple[str, ...]
    exact_terms: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    answer_target: str = ""
    transformer_used: bool = False

    def to_dict(self) -> dict:
        value = asdict(self)
        value["entities"] = list(self.entities)
        value["variants"] = list(self.variants)
        value["exact_terms"] = list(self.exact_terms)
        value["objects"] = list(self.objects)
        return value


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def build_query_plan(query: str) -> QueryPlan:
    normalized = normalize_text(query)
    folded = normalized.casefold()
    intent = "ocr" if any(value in folded for value in OCR_MARKERS) else (
        "visual" if any(value in folded for value in VISUAL_MARKERS) else (
            "asr" if any(value in folded for value in ASR_MARKERS) else "visual"))
    quoted = re.findall(r'["“”]([^"“”]{2,})["“”]', normalized)
    uppercase, run = [], []
    for token in re.findall(r"[\wÀ-ỹĐđ]+", normalized):
        if token.isdigit() or (len(token) >= 2 and token.isupper()):
            run.append(token)
        else:
            if run and any(part.isalpha() for part in run):
                uppercase.append(" ".join(run))
            run = []
    if run and any(part.isalpha() for part in run):
        uppercase.append(" ".join(run))
    marker_capture = re.search(
        r"(?:dòng chữ|chữ|logo|tiêu đề|biển (?:báo|cảnh báo)|ghi)\s+(.+?)(?:[,.]|$)",
        normalized, flags=re.IGNORECASE)
    marker_value = marker_capture.group(1).strip(' "“”') if marker_capture else ""
    ocr_parts = quoted + uppercase
    if not ocr_parts and marker_value:
        ocr_parts.append(marker_value)
    ocr_query = " ".join(dict.fromkeys(part.strip(' "“”') for part in ocr_parts if len(part.strip()) >= 2))
    visual_query = normalized
    if marker_capture:
        visual_query = (normalized[:marker_capture.start()] + normalized[marker_capture.end():]).strip(" ,.")
    if not visual_query:
        visual_query = normalized
    asr_query = normalized if intent == "asr" else ""
    entities = tuple(key for key in ENGLISH_GLOSSES if key in folded)
    english = visual_query
    for phrase in sorted(entities, key=len, reverse=True):
        english = re.sub(re.escape(phrase), ENGLISH_GLOSSES[phrase], english, flags=re.IGNORECASE)
    variants = [visual_query]
    if english.casefold() != visual_query.casefold():
        variants.append(english)
    if entities:
        variants.append(", ".join(ENGLISH_GLOSSES[value] for value in entities))
    exact_terms = tuple(dict.fromkeys([
        *quoted,
        *uppercase,
        *re.findall(r"(?<!\d)\d{2,4}(?!\d)", normalized),
    ]))
    objects = tuple(dict.fromkeys(english_name for vietnamese, english_name in OBJECT_GLOSSES.items()
                                  if vietnamese in folded))
    answer_target = ""
    if any(value in folded for value in ("đồng hồ", "mấy giờ", "thời gian")):
        answer_target = "clock_time"
    elif any(value in folded for value in ("chữ gì", "ghi gì", "nội dung gì")):
        answer_target = "visible_text"
    elif "bao nhiêu" in folded:
        answer_target = "count_or_number"
    return QueryPlan(query, normalized, intent, visual_query, ocr_query, asr_query,
                     entities, tuple(dict.fromkeys(variants)), exact_terms, objects,
                     answer_target, False)
