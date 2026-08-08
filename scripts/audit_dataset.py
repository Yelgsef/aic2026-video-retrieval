from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.audit import audit_dataset
from aic_pipeline.paths import PipelinePaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit AIC video/keyframe/CLIP/object artifacts and build manifests."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="AIC project root containing archive/ and workspace/.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(
        PipelinePaths.from_root(args.root),
        progress_every=args.progress_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
