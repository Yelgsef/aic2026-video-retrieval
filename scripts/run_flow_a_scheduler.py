from __future__ import annotations

from pathlib import Path

from run_dataset_scheduler import DEPENDENCY, main as run_scheduler


ROOT = Path(__file__).resolve().parents[1]
FLOW_A_STAGES = (
    "omnishot",
    "sampling",
    "siglip-base",
    "asr-large-v3",
    "semantic-scenes",
    "siglip-so400m-final",
)
FLOW_A_DEPENDENCY = {
    stage: DEPENDENCY[stage]
    for stage in FLOW_A_STAGES
}
FLOW_A_DEPENDENCY["semantic-scenes"] = ("siglip-base", "asr-large-v3")


def main() -> None:
    run_scheduler({
        "description": "Crash-resumable scheduler for Flow A preprocessing.",
        "stages": FLOW_A_STAGES,
        "dependencies": FLOW_A_DEPENDENCY,
        "db_path": ROOT / "workspace" / "scheduler" / "flow_a_jobs.sqlite3",
    })


if __name__ == "__main__":
    main()
