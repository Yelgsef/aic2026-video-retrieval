from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ID = "L21_V001"

# Manually authored from 30-second contact sheets. The center timestamp is
# resolved to its containing OmniShotCut interval below.
SEEDS = [
    ("q001", 30, "Cảnh quay từ trên cao một đoạn bờ kênh bị sạt lở sát nhà dân", "visual"),
    ("q002", 90, "Biển cảnh báo sạt lở nguy hiểm, yêu cầu tạm dừng lưu thông", "ocr"),
    ("q003", 120, "Một người đàn ông mặc áo trắng phát biểu trong phòng họp", "visual"),
    ("q004", 180, "Nhóm cán bộ khảo sát hiện trường ngoài trời gần khu dân cư", "visual"),
    ("q005", 210, "Khung cảnh đường hoặc khu dân cư bị che phủ bởi sương mù dày", "visual"),
    ("q006", 360, "Một người đàn ông lớn tuổi đứng gần khu vực ao hoặc mặt nước", "visual"),
    ("q007", 390, "Cận cảnh một con cá lớn đang được xử lý trên bề mặt kim loại", "visual"),
    ("q008", 420, "Đàn trâu hoặc bò đang ăn cỏ trên một cánh đồng rộng", "object"),
    ("q009", 450, "Cận cảnh màn hình máy tính xách tay hiển thị văn bản", "ocr"),
    ("q010", 510, "Một phụ nữ ngồi trong nhà trả lời phỏng vấn", "visual"),
    ("q011", 570, "Công an làm việc với giấy tờ cạnh một người mặc áo sọc", "visual"),
    ("q012", 600, "Cận cảnh cánh tay có nhiều hình xăm", "object"),
    ("q013", 660, "Nhiều xe máy màu đỏ được trưng bày hoặc kiểm tra", "object"),
    ("q014", 810, "Toàn cảnh cảng biển với nhiều container và cần cẩu", "visual"),
    ("q015", 840, "Bờ biển bị bão hoặc nước lớn tàn phá, nhiều người đứng quan sát", "visual"),
    ("q016", 870, "Nữ người dẫn chương trình đứng cạnh màn hình hiển thị cháy rừng", "visual"),
    ("q017", 960, "Dãy nhà nhiều màu nằm dọc một con kênh có thuyền", "visual"),
    ("q018", 1050, "Đàn cá bơi dưới làn nước xanh", "object"),
    ("q019", 1110, "Một người đàn ông đeo kính trả lời phỏng vấn ngoài trời", "visual"),
    ("q020", 1200, "Hiện trường tai nạn hoặc điều tra trên đường phố vào ban đêm", "visual"),
]


def main() -> None:
    shot_path = ROOT / "workspace" / "shots" / f"{VIDEO_ID}.json"
    shot_data = json.loads(shot_path.read_text(encoding="utf-8"))
    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    video = videos.loc[videos["video_id"] == VIDEO_ID].iloc[0]
    fps = float(video["fps"])
    ranges = [(int(start), int(end)) for start, end in shot_data["ranges"]]

    queries = []
    for query_id, seconds, text, query_type in SEEDS:
        center_frame = int(round(seconds * fps))
        matches = [
            (shot_id, start, end)
            for shot_id, (start, end) in enumerate(ranges)
            if start <= center_frame < end
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{query_id}: timestamp matches {len(matches)} shots")
        shot_id, start, end = matches[0]
        queries.append(
            {
                "query_id": query_id,
                "task": "KIS",
                "query_type": query_type,
                "query_text_vi": text,
                "video_id": VIDEO_ID,
                "annotation_source": "manual_contact_sheet_30s",
                "annotation_strength": "weak_shot_interval",
                "observed_center_seconds": seconds,
                "observed_center_frame": center_frame,
                "ground_truth_shot_id": shot_id,
                "ground_truth_start_frame": start,
                "ground_truth_end_frame_exclusive": end,
                "ground_truth_start_seconds": start / fps,
                "ground_truth_end_seconds": end / fps,
            }
        )

    output_dir = ROOT / "workspace" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"queries_{VIDEO_ID}.json"
    parquet_path = output_dir / f"queries_{VIDEO_ID}.parquet"
    payload = {
        "schema_version": "0.1.0",
        "video_id": VIDEO_ID,
        "fps": fps,
        "query_count": len(queries),
        "warning": "Weak labels from 30-second contact sheets; review intervals before model selection.",
        "queries": queries,
    }
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(json_path)
    pd.DataFrame(queries).to_parquet(parquet_path, index=False)
    print(json.dumps({"queries": len(queries), "json": str(json_path), "parquet": str(parquet_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
