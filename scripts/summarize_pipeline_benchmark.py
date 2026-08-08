from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "workspace" / "benchmarks" / "pipeline_benchmark.jsonl"
OUTPUT_JSON = ROOT / "workspace" / "benchmarks" / "benchmark_summary.json"
OUTPUT_MD = ROOT / "workspace" / "benchmarks" / "BENCHMARK_SUMMARY.md"


def main() -> None:
    records = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines()]
    completed = pd.DataFrame(item for item in records if item.get("status") == "completed")
    latest = completed.sort_values("finished_at").drop_duplicates(
        ["video_id", "stage", "configuration"], keep="last"
    )
    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    total_duration = float(videos.duration_seconds.fillna(0).sum())
    stage_rows = []
    for (stage, configuration), group in latest.groupby(["stage", "configuration"]):
        weighted_factor = float(group.elapsed_seconds.sum() / group.video_duration_seconds.sum())
        stage_rows.append(
            {
                "stage": stage,
                "configuration": configuration,
                "videos": len(group),
                "weighted_realtime_factor": weighted_factor,
                "projected_dataset_hours_with_process_startup": total_duration * weighted_factor / 3600,
            }
        )

    by_stage = {row["stage"]: row for row in stage_rows}
    measured_current = sum(
        by_stage[name]["projected_dataset_hours_with_process_startup"]
        for name in ("omnishot", "sampling", "siglip-base", "asr-large-v3")
        if name in by_stage
    )
    so400_final = by_stage.get("siglip-so400m-final")
    so400 = by_stage.get("siglip-so400m")
    # Base-selected keyframes retained 238/1668 samples on the median benchmark.
    final_ratio = 238 / 1668
    projected_so400_final = so400_final["projected_dataset_hours_with_process_startup"] if so400_final else (
        so400["projected_dataset_hours_with_process_startup"] * final_ratio
        if so400 else None
    )
    optimized = measured_current + (projected_so400_final or 0)
    report = {
        "dataset": {
            "logical_videos": len(videos),
            "physical_video_files": int(videos.video_copy_count.sum()),
            "duration_hours": total_duration / 3600,
            "size_gb": float(videos.size_bytes.sum() / 1e9),
        },
        "benchmarked_videos": sorted(latest.video_id.unique().tolist()),
        "stages": stage_rows,
        "production_projection": {
            "pipeline": "OmniShotCut + stride5 + SigLIP2 Base clustering + SO400M final-only + Whisper large-v3",
            "single_rtx4060_hours_including_subprocess_startup_estimate": optimized,
            "single_rtx4060_days_with_25_percent_margin": optimized * 1.25 / 24,
            "four_gpu_ideal_hours": optimized / 4,
            "warning": "Model-persistent workers should be faster; Vast transfer and host I/O are not included.",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# AIC pipeline benchmark",
        "",
        f"- Logical videos: {report['dataset']['logical_videos']}",
        f"- Physical files: {report['dataset']['physical_video_files']}",
        f"- Duration: {report['dataset']['duration_hours']:.2f} hours",
        f"- Size: {report['dataset']['size_gb']:.2f} GB",
        f"- Benchmarked: {', '.join(report['benchmarked_videos'])}",
        "",
        "## Measured stages",
        "",
        "| Stage | Configuration | Videos | Realtime factor | Projected hours |",
        "|---|---|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            f"| {row['stage']} | {row['configuration']} | {row['videos']} | "
            f"{row['weighted_realtime_factor']:.3f} | {row['projected_dataset_hours_with_process_startup']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Recommended production path",
            "",
            "`OmniShotCut -> stride 5 -> SigLIP2 Base clustering -> SO400M final keyframes -> Whisper large-v3`",
            "",
            f"Projected single RTX 4060 time: {optimized:.1f} hours before margin.",
            f"With 25% operational margin: {optimized * 1.25 / 24:.1f} days.",
            f"Ideal four-GPU compute time: {optimized / 4:.1f} hours, excluding upload and I/O contention.",
            "",
            "Whisper Turbo was not benchmarked because its model is not present in the local cache.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
