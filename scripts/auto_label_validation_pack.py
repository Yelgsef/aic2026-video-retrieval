from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

from aic_pipeline.router_chat import RouterChatClient


ROOT = Path(__file__).resolve().parents[1]


class ProposedLabel(BaseModel):
    query_text_vi: str = Field(description="One discriminative Vietnamese retrieval query grounded in visible evidence")
    query_type: str = Field(description="Exactly KIS or QA")
    answer: str = Field(description="Short visible answer for QA; empty string for KIS")
    selected_frame_label: str = Field(description="Exactly one tile label such as F1, F2, F3")
    evidence_vi: str = Field(description="Concise visible evidence supporting the query and answer")
    confidence: float = Field(ge=0, le=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate resumable Gemini pseudo-labels for a validation review pack.")
    parser.add_argument("--pack", type=Path, default=ROOT / "workspace" / "validation" / "review_pack_50")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--requests-per-minute", type=float, default=4.5)
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    args.model = args.model or RouterChatClient(ROOT).model
    table = pd.read_csv(args.pack / "annotations.csv", keep_default_na=False)
    table["accepted_frame_id"] = table["accepted_frame_id"].astype(object)
    table["verified"] = table["verified"].astype(object)
    if args.limit:
        table = table.head(args.limit).copy()
    scene_map = {}
    for video_id in table.video_id.unique():
        scenes = json.loads((ROOT / "workspace" / "semantic_scenes" / video_id / "scenes.json").read_text(encoding="utf-8"))["scenes"]
        scene_map.update({str(scene["scene_id"]): scene for scene in scenes})
    raw_dir = args.pack / "auto_labels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    rate_lock = threading.Lock()
    last_request = [0.0]

    def client():
        if not hasattr(local, "client"):
            local.client = RouterChatClient(ROOT, timeout_seconds=120)
        return local.client

    def produce(index: int, row: pd.Series) -> tuple[int, dict]:
        cache = raw_dir / f"{row.query_id}.json"
        if cache.exists():
            return index, json.loads(cache.read_text(encoding="utf-8"))
        scene = scene_map[str(row.scene_id)]
        reps = scene["representative_frames"]
        mapping = {f"F{position + 1}": int(rep["frame_id"]) for position, rep in enumerate(reps)}
        requested_type = "QA" if (index + 1) % 4 == 0 else "KIS"
        prompt = f"""Create one {requested_type} pseudo-label for AIC video retrieval from this chronological contact sheet.
Rules:
- Use only clearly visible evidence, never infer identity, place, or facts not shown.
- selected_frame_label must be one of: {', '.join(mapping)}.
- Avoid queries based only on TV watermark, channel logo, clock, subtitles, or generic news graphics.
- Prefer distinctive people, clothing, objects, actions, setting, composition, or readable event-specific text.
- KIS: describe a concrete event/frame and leave answer empty.
- QA: include a question whose short answer is directly visible in the selected frame.
- The query must identify the scene, not reveal the answer for QA.
- Write Vietnamese naturally. Set confidence below 0.7 if evidence is ambiguous.
"""
        image_path = args.pack / "contact_sheets" / f"{str(row.scene_id).replace(':', '_')}.jpg"
        with Image.open(image_path) as image:
            image_copy = image.copy()
        response = None
        for attempt in range(args.max_retries):
            try:
                with rate_lock:
                    interval = 60.0 / args.requests_per_minute
                    delay = max(0.0, last_request[0] + interval - time.monotonic())
                    if delay:
                        time.sleep(delay)
                    last_request[0] = time.monotonic()
                response = client().json_response(
                    prompt, ProposedLabel, image_copy, model=args.model,
                    temperature=0.1, max_tokens=768)
                break
            except Exception as exc:
                if attempt + 1 >= args.max_retries or not any(code in str(exc) for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                    raise
                time.sleep(min(60, 5 * (attempt + 1)))
        if response is None:
            raise RuntimeError("Gemini returned no response")
        label = ProposedLabel.model_validate_json(response.text).model_dump()
        frame_label = label["selected_frame_label"].strip().upper()
        if frame_label not in mapping:
            raise ValueError(f"Invalid frame label {frame_label} for {row.scene_id}")
        label.update({"query_id": row.query_id, "scene_id": row.scene_id, "video_id": row.video_id,
                      "accepted_frame_id": mapping[frame_label], "model": args.model,
                      "human_verified": False})
        temporary = cache.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(cache)
        return index, label

    labels, errors = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(produce, index, row): index for index, row in table.iterrows()}
        for future in as_completed(futures):
            try:
                index, label = future.result()
                labels[index] = label
                print(json.dumps({"done": len(labels), "query_id": label["query_id"],
                                  "confidence": label["confidence"]}, ensure_ascii=False), flush=True)
            except Exception as exc:
                errors.append({"row": int(futures[future]), "error": f"{type(exc).__name__}: {exc}"})
    for index, label in labels.items():
        table.at[index, "query_type"] = label["query_type"]
        table.at[index, "query_text_vi"] = label["query_text_vi"]
        table.at[index, "answer"] = label["answer"]
        table.at[index, "accepted_frame_id"] = label["accepted_frame_id"]
        table.at[index, "verified"] = False
        table.at[index, "notes"] = (f"AI pseudo-label; confidence={label['confidence']:.2f}; "
                                    f"evidence={label['evidence_vi']}")
    output = args.pack / "annotations_autolabeled.csv"
    table.to_csv(output, index=False, encoding="utf-8-sig")
    report = {"requested": len(table), "labeled": len(labels), "failed": len(errors), "errors": errors,
              "output": str(output), "warning": "AI pseudo-labels; human_verified=False"}
    (args.pack / "auto_label_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
