from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


VIETNAMESE_OBJECTS = {
    "biển số": "license plate", "xe máy": "motorcycle", "ô tô": "car",
    "mũ bảo hiểm": "helmet", "quai mũ": "helmet strap", "áo sơ mi": "shirt",
    "áo": "shirt", "kính": "glasses", "micro": "microphone", "điện thoại": "phone",
    "máy tính bảng": "tablet", "tấm bìa": "paper card", "giấy": "paper",
    "đang cầm": "handheld object", "ghi chép": "hand writing on paper", "viết": "hand writing on paper",
    "con vật": "animal", "động vật": "animal",
    "cá": "fish", "rùa": "turtle", "chó": "dog", "mèo": "cat", "chim": "bird",
    "bốt bảo vệ": "guard booth", "chốt bảo vệ": "guard booth", "khúc gỗ": "wood log",
    "thuyền": "boat", "biển báo": "sign",
}


def grounding_phrases(question: str) -> str:
    lowered = question.casefold()
    phrases = []
    for vietnamese, english in sorted(VIETNAMESE_OBJECTS.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<!\w){re.escape(vietnamese)}(?!\w)", lowered) and english not in phrases:
            phrases.append(english)
    return (". ".join(phrases) + ".") if phrases else ""


class GroundingDinoCropper:
    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Grounding DINO's text enhancer produces FP32 activations in current
        # Transformers releases; forcing the full model to FP16 causes mixed-
        # dtype linear/attention failures.
        self.dtype = torch.float32
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(model_dir), dtype=self.dtype, local_files_only=True).to(self.device).eval()

    def crop(self, images: list[Image.Image], question: str, maximum: int = 4,
             box_threshold: float = 0.22, text_threshold: float = 0.18) -> list[dict[str, Any]]:
        if not images or maximum <= 0:
            return []
        if re.search(r"\b(hướng nào|phía nào|ở đâu)\b", question.casefold()):
            return []
        text = grounding_phrases(question)
        if not text:
            return []
        inputs = self.processor(images=images, text=[text] * len(images), padding=True,
                                return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = [(image.height, image.width) for image in images]
        detections = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=target_sizes)
        candidates = []
        for image_index, (image, result) in enumerate(zip(images, detections)):
            labels = result.get("text_labels", result["labels"])
            for score, label, box in zip(result["scores"], labels, result["boxes"]):
                x1, y1, x2, y2 = [float(value) for value in box]
                pad_x, pad_y = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
                left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
                right, bottom = min(image.width, int(x2 + pad_x + 0.999)), min(image.height, int(y2 + pad_y + 0.999))
                if right - left < 8 or bottom - top < 8:
                    continue
                candidates.append({"image_index": image_index,
                                   "label": re.sub(r"\s+", " ", str(label)).strip(),
                                   "score": float(score), "box": [x1, y1, x2, y2],
                                   "image": image.crop((left, top, right, bottom))})
        candidates.sort(key=lambda item: item["score"], reverse=True)
        output = []
        for candidate in candidates:
            if any(candidate["image_index"] == existing["image_index"]
                   and candidate["label"] == existing["label"] for existing in output):
                continue
            output.append(candidate)
            if len(output) >= maximum:
                break
        return output
