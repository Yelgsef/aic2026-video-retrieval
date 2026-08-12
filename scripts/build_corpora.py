from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.objects_metadata import build_metadata_corpus, build_object_corpus
from aic_pipeline.paths import PipelinePaths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized object and video metadata corpora."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--object-min-confidence", type=float, default=0.15)
    parser.add_argument("--skip-objects", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    args = parser.parse_args()
    paths = PipelinePaths.from_root(args.root)

    reports = {}
    if not args.skip_objects:
        reports["objects"] = build_object_corpus(
            paths,
            min_confidence=args.object_min_confidence,
        )
    if not args.skip_metadata:
        reports["metadata"] = build_metadata_corpus(paths)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
