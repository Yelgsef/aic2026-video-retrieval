from __future__ import annotations

import json

from aic_pipeline.object_spatial import parse_object_constraints, structured_object_scores


def test_parse_count_and_normalized_position() -> None:
    values = parse_object_constraints("Hai ngôi nhà bên trái, ba cái cây ở giữa")
    assert {row["label"] for row in values} == {"house", "tree"}
    house = next(row for row in values if row["label"] == "house")
    tree = next(row for row in values if row["label"] == "tree")
    assert house["count"] == 2 and "left" in house["positions"]
    assert tree["count"] == 3 and "center" in tree["positions"]


def test_structured_score_rewards_matching_count_and_position() -> None:
    matching = {"objects": {"house": {"max_count": 2, "positions": ["left"]}}}
    wrong = {"objects": {"house": {"max_count": 1, "positions": ["right"]}}}
    scores, _ = structured_object_scores(
        "hai ngôi nhà bên trái", [json.dumps(matching), json.dumps(wrong)])
    assert scores[0] > scores[1]
