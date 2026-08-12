from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ID = "L21_V001"
ENCODER_ID = "timm_beit3_base_patch16_224_pt"


def main() -> None:
    queries = pd.read_parquet(ROOT / "workspace" / "validation" / f"queries_{VIDEO_ID}.parquet")
    samples = pd.read_parquet(ROOT / "workspace" / "samples" / f"{VIDEO_ID}.parquet")
    final = pd.read_parquet(
        ROOT / "workspace" / "keyframes" / ENCODER_ID / f"{VIDEO_ID}.parquet"
    )
    rows = []
    for query in queries.itertuples(index=False):
        start = int(query.ground_truth_start_frame)
        end = int(query.ground_truth_end_frame_exclusive)
        sample_hits = samples[(samples.frame_id >= start) & (samples.frame_id < end)]
        final_hits = final[(final.frame_id >= start) & (final.frame_id < end)]
        rows.append(
            {
                "query_id": query.query_id,
                "ground_truth_shot_id": int(query.ground_truth_shot_id),
                "sample_hit_count": len(sample_hits),
                "final_hit_count": len(final_hits),
                "sample_covered": bool(len(sample_hits)),
                "final_covered": bool(len(final_hits)),
                "nearest_final_frame": (
                    int(final_hits.iloc[(final_hits.frame_id - query.observed_center_frame).abs().argmin()].frame_id)
                    if len(final_hits) else None
                ),
            }
        )
    results = pd.DataFrame(rows)
    output = ROOT / "workspace" / "validation" / f"coverage_{VIDEO_ID}.parquet"
    results.to_parquet(output, index=False)
    report = {
        "video_id": VIDEO_ID,
        "query_count": len(results),
        "sampling_interval_coverage": float(results.sample_covered.mean()),
        "final_interval_coverage": float(results.final_covered.mean()),
        "mean_sample_hits_per_query": float(results.sample_hit_count.mean()),
        "mean_final_hits_per_query": float(results.final_hit_count.mean()),
        "uncovered_final_queries": results.loc[~results.final_covered, "query_id"].tolist(),
        "output": str(output),
        "limitation": "Coverage only; this does not measure text-to-frame retrieval ranking.",
    }
    report_path = ROOT / "workspace" / "validation" / f"coverage_{VIDEO_ID}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
