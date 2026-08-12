from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_l21_v001_artifact_counts_match() -> None:
    video_id = "L21_V001"
    mapping = pd.read_csv(ROOT / "archive" / "map-keyframes" / f"{video_id}.csv")
    features = np.load(
        ROOT / "archive" / "clip-features-32" / f"{video_id}.npy",
        mmap_mode="r",
    )
    keyframe_dir = ROOT / "archive" / "keyframes" / "L21" / video_id
    images = list(keyframe_dir.glob("*.jpg"))

    assert len(mapping) == len(images) == features.shape[0]
    assert mapping["frame_idx"].is_monotonic_increasing
    assert mapping.iloc[0]["frame_idx"] == 0
