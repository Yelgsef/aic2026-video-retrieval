from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    index: Path = Path(os.environ.get("AIC_INDEX_DIR", ROOT / "workspace/offline_index"))
    milvus_uri: Path | None = (Path(os.environ["MILVUS_URI"])
                               if os.environ.get("MILVUS_URI") else None)
    collection_prefix: str = os.environ.get("MILVUS_COLLECTION_PREFIX", "aic")
    host: str = os.environ.get("AIC_HOST", "127.0.0.1")
    port: int = int(os.environ.get("AIC_PORT", "8765"))
    device: str | None = os.environ.get("AIC_DEVICE")
    frame_cache: Path = Path(os.environ.get("AIC_FRAME_CACHE", ROOT / "workspace/ui_frame_cache"))
    warmup: bool = os.environ.get("AIC_WARMUP", "1").lower() not in {"0", "false", "no"}


settings = Settings()
