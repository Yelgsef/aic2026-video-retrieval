from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd


LABEL_ALIASES = {
    "person": ("person", "man", "woman", "boy", "girl", "nguoi", "nguoi dan ong", "nguoi phu nu", "co be"),
    "house": ("house", "home", "ngoi nha", "nha"),
    "building": ("building", "toa nha"),
    "tree": ("tree", "plant", "cay", "cay xanh"),
    "flower": ("flower", "bong hoa", "hoa"),
    "vehicle": ("vehicle", "phuong tien"),
    "car": ("car", "automobile", "xe o to", "o to", "xe hoi"),
    "motorcycle": ("motorcycle", "motorbike", "xe may"),
    "bicycle": ("bicycle", "bike", "xe dap"),
    "airplane": ("airplane", "plane", "may bay"),
    "boat": ("boat", "ship", "thuyen", "tau"),
    "bird": ("bird", "chim"), "fish": ("fish", "ca"),
    "animal": ("animal", "con vat", "dong vat"),
    "table": ("table", "desk", "ban", "cai ban"),
    "door": ("door", "cua", "canh cua"),
    "window": ("window", "cua so", "o cua so"),
    "poster": ("poster", "ap phich", "tam ap phich"),
    "helmet": ("helmet", "mu bao hiem"),
    "sculpture": ("sculpture", "statue", "buc tuong", "tuong"),
}
DETECTION_ALIASES = {alias: canonical for canonical, aliases in LABEL_ALIASES.items() for alias in aliases}
NUMBER_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6,
                "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
POSITION_VI = {"ben trai": "left", "phia trai": "left", "ben phai": "right",
               "phia phai": "right", "o giua": "center", "giua": "center",
               "phia tren": "top", "ben tren": "top", "phia duoi": "bottom",
               "ben duoi": "bottom"}


def fold_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold()).replace("đ", "d")
    return " ".join("".join(c if c.isalnum() else " " for c in text
                             if not unicodedata.combining(c)).split())


def normalize_label(value: str) -> str:
    folded = fold_text(value)
    return DETECTION_ALIASES.get(folded, folded)


def spatial_bins(x: float, y: float) -> tuple[str, str]:
    horizontal = "left" if x < 0.36 else "right" if x > 0.64 else "center"
    vertical = "top" if y < 0.36 else "bottom" if y > 0.64 else "middle"
    return horizontal, vertical


def aggregate_scene_objects(detections: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Summarize per-frame boxes without double-counting an object across frames."""
    if detections.empty:
        return {}
    table = detections.loc[detections.confidence.ge(0.20)].copy()
    table["canonical"] = table.label.astype(str).map(normalize_label)
    table["cx"] = (table.x1_norm + table.x2_norm) / 2
    table["cy"] = (table.y1_norm + table.y2_norm) / 2
    output: dict[int, dict[str, Any]] = {}
    for scene_index, scene in table.groupby("scene_index"):
        objects: dict[str, dict[str, Any]] = {}
        for label, group in scene.groupby("canonical"):
            per_frame = group.groupby("frame_id").size()
            bins = Counter(position for x, y in zip(group.cx, group.cy)
                           for position in spatial_bins(float(x), float(y)))
            objects[str(label)] = {"max_count": int(per_frame.max()),
                                   "positions": [name for name, _ in bins.most_common()]}
        relations: Counter[str] = Counter()
        for _, frame in scene.groupby("frame_id"):
            selected = frame.sort_values("confidence", ascending=False).head(12)
            records = list(selected.itertuples(index=False))
            for index, left in enumerate(records):
                for right in records[index + 1:]:
                    if left.canonical == right.canonical:
                        continue
                    dx, dy = float(left.cx - right.cx), float(left.cy - right.cy)
                    if abs(dx) >= 0.15:
                        relation = "right_of" if dx > 0 else "left_of"
                        relations[f"{left.canonical} {relation} {right.canonical}"] += 1
                    if abs(dy) >= 0.15:
                        relation = "below" if dy > 0 else "above"
                        relations[f"{left.canonical} {relation} {right.canonical}"] += 1
                    if float(np.hypot(dx, dy)) < 0.25:
                        relations[f"{left.canonical} near {right.canonical}"] += 1
        output[int(scene_index)] = {"objects": objects,
                                    "relations": [value for value, _ in relations.most_common(40)]}
    return output


def object_spatial_text(metadata: dict[str, Any]) -> str:
    parts = []
    for label, data in metadata.get("objects", {}).items():
        count = int(data.get("max_count", 0))
        positions = " ".join(str(value) for value in data.get("positions", []))
        parts.append(f"object {label}; count {label} {count}; position {label} {positions}")
    parts.extend(f"relation {relation}" for relation in metadata.get("relations", []))
    return "; ".join(parts)


def parse_object_constraints(query: str) -> list[dict[str, Any]]:
    folded = fold_text(query)
    matches: list[tuple[int, int, str]] = []
    for canonical, aliases in LABEL_ALIASES.items():
        for alias in sorted((fold_text(value) for value in aliases), key=len, reverse=True):
            for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", folded):
                if not any(start <= match.start() < end for start, end, _ in matches):
                    matches.append((match.start(), match.end(), canonical))
    constraints = []
    for start, end, label in sorted(matches):
        context = folded[max(0, start - 24):min(len(folded), end + 32)]
        before = folded[max(0, start - 18):start].split()
        count = None
        for token in reversed(before[-4:]):
            if token.isdigit():
                count = int(token); break
            if token in NUMBER_WORDS:
                count = NUMBER_WORDS[token]; break
        positions = list(dict.fromkeys(value for phrase, value in POSITION_VI.items() if phrase in context))
        constraints.append({"label": label, "count": count, "positions": positions})
    return constraints


def structured_object_scores(query: str, metadata_values: list[Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    constraints = parse_object_constraints(query)
    scores = np.zeros(len(metadata_values), dtype=np.float32)
    if not constraints:
        return scores, constraints
    import json
    for row, raw in enumerate(metadata_values):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        metadata = json.loads(raw) if isinstance(raw, str) else raw
        values = []
        for constraint in constraints:
            found = metadata.get("objects", {}).get(constraint["label"])
            if not found:
                values.append(0.0); continue
            components = [1.0]
            if constraint["count"] is not None:
                actual, expected = int(found.get("max_count", 0)), int(constraint["count"])
                components.append(1.0 if actual == expected else 0.7 if actual >= expected else actual / expected)
            if constraint["positions"]:
                observed = set(found.get("positions", []))
                components.append(sum(position in observed for position in constraint["positions"])
                                  / len(constraint["positions"]))
            values.append(float(np.mean(components)))
        scores[row] = float(np.mean(values))
    return scores, constraints
