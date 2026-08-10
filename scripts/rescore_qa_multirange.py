from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore saved QA predictions against multiple accepted ranges.")
    parser.add_argument("--predictions", type=Path, default=ROOT / "workspace/validation/qa_end_to_end_subset10_refined.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "workspace/validation/qa_multirange_subset10.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/validation/qa_end_to_end_subset10_multirange.json")
    args = parser.parse_args()
    report = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = {x["query_id"]: x for x in
              (json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip())}
    for row in report["rows"]:
        ranges = labels[row["query_id"]]["accepted_ranges"]
        row["single_range_frame_hit"] = bool(row["frame_hit"])
        row["frame_hit"] = any(str(row.get("predicted_video_id")) == str(item["video_id"])
                               and int(item["start_frame"]) <= int(row.get("predicted_frame_id", -1))
                               < int(item["end_frame_exclusive"]) for item in ranges)
        row["accepted_range_count"] = len(ranges)
    count = len(report["rows"])
    report.update({"multi_range_labels": True,
                   "frame_hit_rate": sum(x["frame_hit"] for x in report["rows"]) / count,
                   "joint_accuracy": sum(x["frame_hit"] and x["answer_match"] for x in report["rows"]) / count})
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("count", "frame_hit_rate", "answer_match_rate", "joint_accuracy")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
