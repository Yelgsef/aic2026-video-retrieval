from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def metric(rows: list[dict], query_type: str) -> tuple[float, float]:
    chosen = [x for x in rows if x.get("query_type") == query_type]
    mrr = sum(1 / x["rank"] if x.get("rank") else 0 for x in chosen) / max(1, len(chosen))
    r5 = sum(bool(x.get("rank") and x["rank"] <= 5) for x in chosen) / max(1, len(chosen))
    return mrr, r5


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a fusion ablation with paired bootstrap safeguards.")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("workspace/validation/fusion_selection.json"))
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    baseline = reports[0]
    summary = []
    rng = random.Random(2026)
    base_by_id = {x["query_id"]: x for x in baseline["rows"]}
    for path, report in zip(args.reports, reports):
        row = {"report": str(path), "config": {k: report.get(k) for k in
               ("fusion_mode", "field_aware", "openclip")}}
        for task in ("KIS", "QA"):
            row[task] = dict(zip(("mrr", "recall_at_5"), metric(report["rows"], task)))
        paired = [(base_by_id[x["query_id"]], x) for x in report["rows"] if x["query_id"] in base_by_id]
        deltas = []
        for _ in range(args.bootstrap):
            sample = [paired[rng.randrange(len(paired))] for _ in paired]
            deltas.append(sum((1/b["rank"] if b.get("rank") else 0) -
                              (1/a["rank"] if a.get("rank") else 0) for a, b in sample) / len(sample))
        row["mrr_delta_ci95"] = [sorted(deltas)[int(.025*len(deltas))],
                                 sorted(deltas)[int(.975*len(deltas))-1]]
        summary.append(row)
    # The service routes tasks independently, so do not reject a KIS specialist
    # because it is weaker on QA (or vice versa).
    winners = {task: max(summary, key=lambda x: (x[task]["mrr"], x[task]["recall_at_5"]))
               for task in ("KIS", "QA")}
    payload = {"pseudo_labels": True, "baseline": summary[0], "candidates": summary,
               "selected_by_task": winners,
               "warning": "Re-run on human-verified strong labels before production."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
