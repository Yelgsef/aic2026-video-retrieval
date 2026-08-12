from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.paths import PipelinePaths
from aic_pipeline.sampling import build_video_sampling_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stride sampling manifest from OmniShotCut ranges."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--shots", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--coverage-anchor-interval", type=int, default=0,
                        help="Hard-retain one global frame per N-frame window; 0 disables it.")
    parser.add_argument(
        "--stride-only", action="store_true",
        help="Use only shot-local stride frames; omit start/middle/end and boundary anchors.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    report = build_video_sampling_manifest(
        PipelinePaths.from_root(args.root),
        video_id=args.video_id,
        shot_json_path=args.shots,
        stride=args.stride,
        boundary_radius=args.boundary_radius,
        include_anchors=not args.stride_only,
        coverage_anchor_interval=args.coverage_anchor_interval,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
