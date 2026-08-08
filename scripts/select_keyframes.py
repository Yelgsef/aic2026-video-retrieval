from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.keyframe_selection import select_video_keyframes
from aic_pipeline.paths import PipelinePaths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive shot-local clustering and conservative redundancy elimination."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--encoder-id", default="openclip_vit_b32_openai_quickgelu")
    parser.add_argument("--k-cap", type=int, default=12)
    parser.add_argument("--redundancy-cosine", type=float, default=0.985)
    parser.add_argument("--temporal-window-frames", type=int, default=15)
    parser.add_argument("--minimum-frames-per-shot", type=int, default=3)
    parser.add_argument(
        "--protected-anchor-policy", choices=("none", "core", "all"), default="core",
        help="none keeps only medoids; core protects shot start/middle/end; all also protects boundary neighbors.",
    )
    parser.add_argument(
        "--adjacent-shot-cosine", type=float, default=None,
        help="Optionally remove near-duplicate medoids across consecutive shots.",
    )
    parser.add_argument(
        "--global-within-shot-cosine", type=float, default=None,
        help="Optionally compare all medoids within a shot, regardless of temporal distance.",
    )
    parser.add_argument(
        "--single-cluster-max-distance", type=float, default=0.08,
        help="Use k=1 when mean cosine distance to the shot center is at most this value.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = select_video_keyframes(
        PipelinePaths.from_root(PROJECT_ROOT),
        video_id=args.video_id,
        encoder_id=args.encoder_id,
        k_cap=args.k_cap,
        redundancy_cosine=args.redundancy_cosine,
        temporal_window_frames=args.temporal_window_frames,
        minimum_frames_per_shot=args.minimum_frames_per_shot,
        protected_anchor_policy=args.protected_anchor_policy,
        adjacent_shot_cosine=args.adjacent_shot_cosine,
        global_within_shot_cosine=args.global_within_shot_cosine,
        single_cluster_max_distance=args.single_cluster_max_distance,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
