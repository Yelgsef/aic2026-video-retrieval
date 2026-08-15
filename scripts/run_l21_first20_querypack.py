from __future__ import annotations

import json
import argparse
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8766"
QUERIES = [
    ("KIS", "kis_01", "Đoạn clip về các phóng sự. Có một đoạn quay chiếc xe máy cày trên đồng lúa xanh. Có một phóng sự về giảm 50% lệ phí trước bạ cho ô tô lắp ráp trong nước. Một đoạn quay về các vận động viên xe đạp."),
    ("KIS", "kis_02", "Một đoạn clip phóng sự về các nghệ nhân truyền thống. Mở đầu clip là hai người phụ nữ lớn tuổi ngồi đan chiếu. Đoạn clip kết thúc với cảnh quay nguyên liệu làm chiếu là bó sợi nhiều màu sặc sỡ."),
    ("KIS", "kis_03", "Đoạn clip phóng sự về tình trạng mưa kỉ lục ở một nước châu Âu. Đoạn clip gồm hình ảnh dòng nước lũ màu nâu đất cuồn cuộn, phía sau là những căn nhà dân."),
    ("QA", "qa_01", "Đoạn clip về diễn biến dịch đậu mùa khỉ châu Phi. Khung cảnh gồm có những đứa trẻ châu Phi ngồi trên đất, có một đứa trẻ đứng bên cạnh. Phía sau là những túp lều y tế màu trắng dày đặc. Hỏi dòng chữ to màu xanh da trời được in trên những túp lều là gì?"),
    ("QA", "qa_02", "Đoạn video về phóng sự một hang động ở Bosnia. Hỏi tên của du khách người Mỹ được phỏng vấn tên gì?"),
    ("QA", "qa_03", "Đoạn clip về mùa vụ ở tỉnh Đồng Tháp. Trong đoạn clip có rất nhiều đàn vịt. Lấy khoảnh khắc không có con vịt nào trong khung hình. Hỏi người đàn ông đứng bên trái khung hình mặc áo màu gì?"),
    ("TRAKE", "trake_01", "Đoạn clip về hình ảnh trong nhà máy sản xuất chip; (1) Một người công nhân mặc đồ bảo hộ trắng toàn thân đi giữa hai hàng máy thiết bị màu đen; (2) Khoảnh khắc khi logo Intel ở trước một tòa nhà nằm gọn trong khung hình; (3) Khoảnh khắc có hai người công nhân mặc đồ bảo hộ trắng, phía sau có ánh sáng tím."),
    ("TRAKE", "trake_02", "Đoạn phóng sự về bảo tồn rùa biển; (1) Khoảnh khắc có 3 em học sinh mặc áo đỏ đứng gần khung hình, trên tay cầm hai chiếc tô màu nâu đen; (2) Khoảnh khắc camera quay một tấm áp phích in nhiều hình rùa, và các thông tin khoa học về các loài rùa này; (3) Khoảnh khắc người đàn ông mặc áo xanh dương xúc nắm cát thứ hai bằng tay ra khỏi hố."),
]


def call(endpoint: str, payload: dict) -> dict:
    request = urllib.request.Request(
        BASE + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def main() -> None:
    global BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--base-url", default=BASE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "workspace/validation/l21_first20_querypack_results")
    args = parser.parse_args()
    BASE = args.base_url.rstrip("/")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for task, query_id, query in QUERIES:
        if args.only and query_id not in args.only:
            continue
        started = time.perf_counter()
        if task == "KIS":
            endpoint = "/query"
            payload = {"query": query, "top_k": 100, "candidate_k": 300,
                       "rerank_k": 30, "query_transformer": False,
                       "moment_refinement": False}
        elif task == "QA":
            endpoint = "/answer"
            payload = {"query": query, "task": "QA", "top_k": 100,
                       "candidate_k": 60, "rerank_k": 30, "vqa_candidates": 24,
                       "query_transformer": False}
        else:
            endpoint = "/trake"
            payload = {"query": query, "task": "TRAKE", "query_transformer": False}
        try:
            result = call(endpoint, payload)
            status, error = "ok", None
        except Exception as exc:
            result, status, error = {}, "error", f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        record = {"query_id": query_id, "task": task, "query": query,
                  "status": status, "error": error, "elapsed_seconds": round(elapsed, 3),
                  "result": result}
        (output / f"{query_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(record)
        print(json.dumps({"query_id": query_id, "task": task, "status": status,
                          "elapsed_seconds": round(elapsed, 3), "error": error},
                         ensure_ascii=False), flush=True)
    (output / "all_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
