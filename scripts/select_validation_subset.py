from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a deterministic diverse validation subset.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/validation/subset20.json")
    args = parser.parse_args()
    table = pd.read_parquet(ROOT / "workspace/manifests/videos.parquet")
    table = table.loc[table.video_readable].copy()
    table["group"] = table.video_id.str.extract(r"^(L\d+)")
    table["duration_bin"] = pd.qcut(table.duration_seconds.rank(method="first"), 4, labels=False)
    selected, used = [], set()
    # Round-robin groups and duration quartiles avoids selecting only one news source.
    for duration_bin in range(4):
        for group in sorted(table.group.dropna().unique()):
            rows = table.loc[(table.group == group) & (table.duration_bin == duration_bin)].sort_values("video_id")
            for row in rows.itertuples(index=False):
                if row.video_id not in used:
                    selected.append(row); used.add(row.video_id); break
            if len(selected) >= args.count:
                break
        if len(selected) >= args.count:
            break
    if len(selected) < args.count:
        for row in table.sort_values("video_id").itertuples(index=False):
            if row.video_id not in used:
                selected.append(row); used.add(row.video_id)
            if len(selected) >= args.count:
                break
    payload = {"schema_version": "0.1.0", "count": len(selected),
               "video_ids": [x.video_id for x in selected],
               "duration_hours": sum(float(x.duration_seconds) for x in selected) / 3600,
               "size_gb": sum(int(x.size_bytes) for x in selected) / 1e9,
               "selection": "round-robin source group x duration quartile"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
