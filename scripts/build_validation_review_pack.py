from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.gemini_batch_scenes import contact_sheet, extract_frames, video_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create contact sheets and a CSV template for strong validation labels.")
    parser.add_argument("--video-id", action="append", required=True)
    parser.add_argument("--scenes-per-video", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / "validation" / "review_pack")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "contact_sheets"
    annotations, cards = [], []
    for video_id in args.video_id:
        source = ROOT / "workspace" / "semantic_scenes" / video_id / "scenes.json"
        scenes = json.loads(source.read_text(encoding="utf-8"))["scenes"]
        count = min(args.scenes_per_video, len(scenes))
        positions = sorted({round(index * (len(scenes) - 1) / max(count - 1, 1)) for index in range(count)})
        selected = [scenes[position] for position in positions]
        frame_ids = {int(rep["frame_id"]) for scene in selected for rep in scene["representative_frames"]}
        decoded = extract_frames(video_path(video_id), frame_ids)
        for scene in selected:
            scene_id = str(scene["scene_id"])
            relative = Path("contact_sheets") / f"{scene_id.replace(':', '_')}.jpg"
            contact_sheet(scene, decoded, args.output / relative)
            query_id = f"strong_{len(annotations) + 1:04d}"
            annotations.append({"query_id": query_id, "query_type": "", "difficulty": "",
                                "query_text_vi": "", "answer": "", "video_id": video_id,
                                "scene_id": scene_id, "start_frame": int(scene["start_frame"]),
                                "end_frame_exclusive": int(scene["end_frame_exclusive"]),
                                "start_time": float(scene["start_time"]), "end_time": float(scene["end_time"]),
                                "accepted_frame_id": "", "verified": False, "notes": ""})
            cards.append(f"<article><h2>{html.escape(query_id)} — {html.escape(scene_id)}</h2>"
                         f"<p>{scene['start_time']:.2f}s–{scene['end_time']:.2f}s | frames "
                         f"{scene['start_frame']}–{scene['end_frame_exclusive'] - 1}</p>"
                         f"<img loading='lazy' src='{relative.as_posix()}'></article>")
    table = pd.DataFrame(annotations)
    table.to_csv(args.output / "annotations.csv", index=False, encoding="utf-8-sig")
    page = """<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>AIC validation review</title>
<style>body{font-family:Arial;max-width:1200px;margin:auto;background:#eee}article{background:white;padding:16px;margin:16px 0}img{max-width:100%;height:auto}h2{margin:0 0 8px}</style>
</head><body><h1>AIC strong validation review pack</h1><p>Điền annotations.csv; mỗi query phải mô tả nội dung nhìn thấy trong đúng khoảng scene.</p>"""
    (args.output / "index.html").write_text(page + "".join(cards) + "</body></html>", encoding="utf-8")
    report = {"videos": args.video_id, "scenes": len(table), "csv": str(args.output / "annotations.csv"),
              "html": str(args.output / "index.html")}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
