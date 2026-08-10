from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic TRAKE pseudo-labels from enriched scenes.")
    parser.add_argument("--video-id", action="append", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/validation/trake_pseudolabels_subset10.jsonl")
    args = parser.parse_args()
    labels = []
    for video_id in args.video_id:
        table = pd.read_parquet(ROOT / "workspace/scene_documents" / video_id / "scenes.parquet")
        table = table.loc[table.visual_summary.fillna("").str.len().ge(20)].sort_values("scene_index").reset_index(drop=True)
        if len(table) < 3:
            continue
        positions = sorted(set(int(round((len(table) - 1) * q)) for q in (0.2, 0.5, 0.8)))
        if len(positions) < 3:
            positions = [0, len(table) // 2, len(table) - 1]
        selected = table.iloc[positions[:3]]
        events = []
        for event_index, row in enumerate(selected.itertuples(index=False)):
            events.append({"event_index": event_index, "description_vi": str(row.visual_summary),
                           "scene_id": str(row.document_id), "accepted_frame_id": int(row.representative_frame_ids[0]),
                           "start_frame": int(row.start_frame), "end_frame_exclusive": int(row.end_frame_exclusive)})
        labels.append({"query_id": f"trake_subset10_{len(labels)+1:04d}", "query_type": "TRAKE",
                       "difficulty": "hard", "query_text_vi": "Tìm chuỗi sự kiện theo thứ tự: " +
                       "; ".join(f"({i+1}) {event['description_vi']}" for i, event in enumerate(events)),
                       "video_id": video_id, "events": events, "verified": False,
                       "notes": "Gemini scene-summary pseudo-label; not human verified"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "labels": len(labels)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
