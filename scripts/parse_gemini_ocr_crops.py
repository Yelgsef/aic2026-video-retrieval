from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Gemini crop OCR into local full-resolution OCR regions.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--responses", type=Path)
    args = parser.parse_args()
    root = ROOT / "workspace/ocr_fullres" / args.video_id
    responses = args.responses or root / "responses.jsonl"
    values, errors = {}, []
    for number, line in enumerate(responses.read_text(encoding="utf-8").splitlines(), 1):
        try:
            envelope = json.loads(line); key = str(envelope["key"])
            candidates = envelope.get("response", envelope).get("candidates") or []
            text = "".join(str(x.get("text", "")) for x in candidates[0]["content"]["parts"])
            value = json.loads(text.removeprefix("```json").removesuffix("```").strip())
            match = re.search(r":f(\d+):r(\d+)$", key)
            if not match:
                raise ValueError("invalid key")
            values[(int(match.group(1)), int(match.group(2)))] = value
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"line": number, "error": str(exc)})
    table = pd.read_parquet(root / "regions.parquet")
    gemini_text, final_text, verified = [], [], []
    for row in table.itertuples(index=False):
        value = values.get((int(row.frame_id), int(row.region_index)))
        accepted = bool(value and value.get("readable") and str(value.get("text", "")).strip())
        gemini_text.append(str(value.get("text", "")) if value else "")
        final_text.append(str(value["text"]).strip() if accepted else str(row.text_raw))
        verified.append(accepted)
    table["gemini_text"], table["final_text"], table["gemini_verified"] = gemini_text, final_text, verified
    table.to_parquet(root / "regions.parquet", index=False)
    report = {"video_id": args.video_id, "responses": len(values), "accepted": sum(verified),
              "errors": errors, "output": str(root / "regions.parquet")}
    (root / "gemini_merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
