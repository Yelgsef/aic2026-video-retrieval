from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified dev/test label split.")
    parser.add_argument("--labels", type=Path, default=ROOT / "workspace/validation/review_pack_50/annotations_autolabeled.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/validation/strong_silver_split")
    parser.add_argument("--dev-fraction", type=float, default=0.6)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        raise ValueError("dev-fraction must be between zero and one")
    labels = pd.read_csv(args.labels).sort_values(["query_type", "video_id", "start_frame", "query_id"])
    parts = {"dev": [], "test": []}
    # Interleave chronologically within every task/video group. This prevents
    # either split from containing only the beginning or end of a video.
    for _, group in labels.groupby(["query_type", "video_id"], sort=True):
        group = group.reset_index(drop=True)
        dev_count = round(len(group) * args.dev_fraction)
        priorities = sorted(range(len(group)), key=lambda i: ((i * 0.61803398875) % 1, i))
        dev_indices = set(priorities[:dev_count])
        parts["dev"].append(group.loc[[i in dev_indices for i in range(len(group))]])
        parts["test"].append(group.loc[[i not in dev_indices for i in range(len(group))]])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.labels), "dev_fraction": args.dev_fraction, "splits": {}}
    for name, groups in parts.items():
        frame = pd.concat(groups, ignore_index=True).sort_values("query_id")
        path = args.output / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        report["splits"][name] = {
            "path": str(path), "count": len(frame),
            "query_types": frame.query_type.value_counts().sort_index().to_dict(),
            "videos": int(frame.video_id.nunique()),
        }
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
