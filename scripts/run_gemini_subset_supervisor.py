from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_IDS = (
    "L21_V001", "L21_V002", "L21_V003", "L21_V005", "L21_V006",
    "L21_V007", "L21_V008", "L21_V009", "L21_V010", "L21_V011",
    "L21_V012", "L21_V013", "L21_V014", "L21_V015", "L21_V016",
    "L21_V017", "L21_V018", "L21_V019", "L21_V021", "L21_V022",
)
LOCAL_STAGES = (
    "omnishot", "sampling", "siglip-base", "siglip-so400m-final",
    "asr-large-v3", "semantic-scenes", "objects-yoloworld", "fullres-ocr",
    "asr-normalize-prepare", "gemini-batch-prepare", "gemini-ocr-crop-prepare",
)


def run(command: list[str], attempts: int = 1) -> str:
    last_error = ""
    for attempt in range(attempts):
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                 encoding="utf-8", errors="replace")
        if process.stdout.strip():
            print(process.stdout.strip(), flush=True)
        if process.returncode == 0:
            return process.stdout
        if process.returncode == 75:
            raise SystemExit(75)
        last_error = (process.stderr or process.stdout).strip()
        print(f"COMMAND_RETRY {attempt + 1}/{attempts}: {last_error[-1000:]}", flush=True)
        time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"Command failed: {' '.join(command)}\n{last_error}")


def scheduler_counts(video_ids: tuple[str, ...]) -> dict[str, int]:
    from run_dataset_scheduler import configuration_hash

    path = ROOT / "workspace/scheduler/jobs.sqlite3"
    connection = sqlite3.connect(path)
    marks = ",".join("?" for _ in video_ids)
    stage_marks = ",".join("?" for _ in LOCAL_STAGES)
    rows = connection.execute(
        f"SELECT status,COUNT(*) FROM jobs WHERE config_hash=? "
        f"AND video_id IN ({marks}) AND stage IN ({stage_marks}) GROUP BY status",
        (configuration_hash(), *video_ids, *LOCAL_STAGES),
    ).fetchall()
    connection.close()
    return {str(status): int(count) for status, count in rows}


def wait_for_local(video_ids: tuple[str, ...], poll_seconds: int) -> None:
    expected = len(video_ids) * len(LOCAL_STAGES)
    while True:
        counts = scheduler_counts(video_ids)
        print(f"LOCAL_PROGRESS {counts.get('completed', 0)}/{expected} {counts}", flush=True)
        if counts.get("failed", 0):
            raise RuntimeError(f"Local scheduler has failed jobs: {counts}")
        if counts.get("completed", 0) == expected:
            return
        time.sleep(poll_seconds)


def parse_state(output: str) -> str:
    for line in reversed(output.splitlines()):
        if "JOB_STATE_" in line and '"state"' in line:
            return line.split('"state"', 1)[1].split('"', 2)[1]
    try:
        start, end = output.find("{"), output.rfind("}") + 1
        return str(json.loads(output[start:end]).get("state", ""))
    except Exception:
        return ""


def prepared_auxiliary(kind: str, video_id: str) -> bool:
    if kind == "asr-normalize":
        path = ROOT / "workspace/asr_normalization" / video_id / "requests.jsonl"
    else:
        path = ROOT / "workspace/ocr_fullres" / video_id / "gemini_requests.jsonl"
    return path.exists() and path.stat().st_size > 0


def process_9router(video_ids: tuple[str, ...], model: str | None, workers: int,
                    force: bool = False, min_interval: float = 0.25) -> None:
    for video_id in video_ids:
        run(["uv", "run", "--project", "environments/gemini", "python",
             "scripts/run_9router_enrichment.py", "--task", "scenes",
             "--video-id", video_id, "--workers", str(workers),
             "--min-interval", str(min_interval)]
            + (["--force"] if force else [])
            + (["--model", model] if model else []), attempts=3)
        for kind in ("asr-normalize", "ocr-crops"):
            if not prepared_auxiliary(kind, video_id):
                print(f"SKIP_EMPTY {kind} {video_id}", flush=True)
                continue
            run(["uv", "run", "--project", "environments/gemini", "python",
                 "scripts/run_9router_enrichment.py", "--task", kind,
                 "--video-id", video_id, "--workers", str(workers),
                 "--min-interval", str(min_interval)]
                + (["--force"] if force else [])
                + (["--model", model] if model else []), attempts=3)


def build_documents_and_index(video_ids: tuple[str, ...], output: Path) -> None:
    for video_id in video_ids:
        run(["uv", "run", "--project", "environments/gemini", "python",
             "scripts/parse_gemini_batch_results.py", "--video-id", video_id])
        asr_response = ROOT / "workspace/asr_normalization" / video_id / "responses.jsonl"
        if asr_response.exists() and asr_response.stat().st_size:
            run(["uv", "run", "--project", "environments/gemini", "python",
                 "scripts/parse_gemini_asr_normalization.py", "--video-id", video_id])
        ocr_response = ROOT / "workspace/ocr_fullres" / video_id / "responses.jsonl"
        if ocr_response.exists() and ocr_response.stat().st_size:
            run(["uv", "run", "--project", "environments/gemini", "python",
                 "scripts/parse_gemini_ocr_crops.py", "--video-id", video_id])
        run(["uv", "run", "--project", "environments/siglip2", "python",
             "scripts/build_unified_scene_documents.py", "--video-id", video_id])

    resolved = output.resolve()
    workspace = (ROOT / "workspace").resolve()
    if workspace not in resolved.parents:
        raise ValueError(f"Index output must be inside workspace: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    command = ["uv", "run", "--project", "environments/siglip2", "python",
               "scripts/build_offline_hybrid_index.py", "--output", str(resolved)]
    for video_id in video_ids:
        command.extend(("--video-id", video_id))
    run(command)
    run(["uv", "run", "--project", "environments/siglip2", "python",
         "scripts/build_bge_m3_scene_index.py", "--documents", str(resolved / "scenes.parquet"),
         "--output", str(resolved), "--batch-size", "16"])
    run(["uv", "run", "--project", "environments/siglip2", "python",
         "scripts/build_openclip_frame_index.py", "--index", str(resolved), "--force"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish 9Router enrichment and index a scheduler subset.")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--model", help="Override NINEROUTER_MODEL from .env")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=0.25,
                        help="Minimum seconds between 9Router request starts per process.")
    parser.add_argument("--force-router", action="store_true",
                        help="Regenerate every 9Router response instead of resuming old output.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "workspace/offline_index_l21_first20_9router")
    parser.add_argument("--shutdown", action="store_true",
                        help="Shut Windows down only after every stage and index build succeeds.")
    parser.add_argument("--shutdown-delay", type=int, default=60)
    args = parser.parse_args()
    video_ids = tuple(args.video_ids or DEFAULT_VIDEO_IDS)
    wait_for_local(video_ids, max(30, args.poll_seconds))
    process_9router(video_ids, args.model, args.workers, args.force_router, args.min_interval)
    build_documents_and_index(video_ids, args.output)
    report = {"status": "complete", "video_ids": list(video_ids),
              "index": str(args.output.resolve()), "model": args.model}
    report_path = ROOT / "workspace/scheduler/l21_first20_supervisor_complete.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.shutdown:
        delay = max(30, int(args.shutdown_delay))
        subprocess.run(
            ["shutdown.exe", "/s", "/t", str(delay), "/c",
             "AIC 9Router pipeline and offline index completed successfully."],
            check=True,
        )
        print(f"SHUTDOWN_SCHEDULED delay_seconds={delay}", flush=True)


if __name__ == "__main__":
    main()
