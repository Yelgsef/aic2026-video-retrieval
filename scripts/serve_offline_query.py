from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Flow C FastAPI service.")
    parser.add_argument("--host", default=os.environ.get("AIC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIC_PORT", "8765")))
    parser.add_argument("--index", type=Path, default=Path(os.environ.get(
        "AIC_INDEX_DIR", ROOT / "workspace/offline_index")))
    parser.add_argument("--milvus-uri", type=Path)
    parser.add_argument("--collection-prefix", default=os.environ.get("MILVUS_COLLECTION_PREFIX", "aic"))
    parser.add_argument("--device", default=os.environ.get("AIC_DEVICE"))
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    os.environ["AIC_HOST"] = args.host
    os.environ["AIC_PORT"] = str(args.port)
    os.environ["AIC_INDEX_DIR"] = str(args.index.resolve())
    os.environ["MILVUS_COLLECTION_PREFIX"] = args.collection_prefix
    if args.milvus_uri:
        os.environ["MILVUS_URI"] = str(args.milvus_uri.resolve())
    if args.device:
        os.environ["AIC_DEVICE"] = args.device
    if args.no_warmup:
        os.environ["AIC_WARMUP"] = "0"

    import uvicorn
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
