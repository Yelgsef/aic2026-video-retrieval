from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from aic_pipeline.query_transformer import LocalQueryTransformer
from aic_pipeline.query_understanding import build_query_plan


queries = [
    "Trong cảnh vận động viên xe đạp mang số áo 84, đồng hồ thời gian ở góc màn hình đang hiển thị bao nhiêu?",
    "Một người phụ nữ mặc áo xanh đứng cạnh xe máy trên đường phố",
    "Biển báo có dòng chữ ROAD CLOSED xuất hiện ở đâu?",
    "Một người đàn ông cầm micro đứng bên trái một chiếc ô tô màu đỏ",
    "Cận cảnh hai trẻ em đang ngồi trước màn hình máy tính",
]
transformer = LocalQueryTransformer(ROOT, ROOT / "workspace/offline_index_subset20")
results, timings = [], []
try:
    for query in queries:
        started = time.perf_counter()
        results.append(transformer.enhance(build_query_plan(query)).to_dict())
        timings.append(round(time.perf_counter() - started, 3))
finally:
    if transformer.server:
        transformer.server.terminate()
print(json.dumps({"timings": timings, "results": results}, ensure_ascii=False, indent=2))
