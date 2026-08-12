from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.clip_features import pack_provided_clip
from aic_pipeline.paths import PipelinePaths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize and concatenate provided CLIP ViT-B/32 features."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    report = pack_provided_clip(
        PipelinePaths.from_root(args.root),
        progress_every=args.progress_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
