from __future__ import annotations

import pandas as pd


def test_aligned_coverage_windows_can_be_audited() -> None:
    selected = pd.DataFrame({"frame_id": [20, 130, 250]})
    missing = []
    for start in range(0, 309, 103):
        end = min(start + 103, 309)
        if not selected.frame_id.between(start, end - 1).any():
            missing.append((start, end))

    assert missing == []


def test_coverage_detects_an_empty_window() -> None:
    selected = pd.DataFrame({"frame_id": [20, 250]})
    missing = []
    for start in range(0, 309, 103):
        end = min(start + 103, 309)
        if not selected.frame_id.between(start, end - 1).any():
            missing.append((start, end))

    assert missing == [(103, 206)]
