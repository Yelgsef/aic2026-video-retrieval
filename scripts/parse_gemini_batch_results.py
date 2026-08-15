from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("summary_vi", "ocr_lines", "people", "objects", "actions", "locations", "events",
          "search_keywords_vi", "uncertainties")


def response_text(payload: dict[str, Any]) -> str:
    response = payload.get("response", payload)
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError(str(payload.get("error") or response.get("error") or "response has no candidates"))
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise ValueError("candidate has no text")
    return text.removeprefix("```json").removesuffix("```").strip()


def normalize(value: dict[str, Any]) -> dict[str, Any]:
    aliases = {"location": "locations", "event": "events"}
    for old, new in aliases.items():
        if old in value and new not in value:
            value[new] = value.pop(old)
    output = {field: value.get(field, "" if field == "summary_vi" else []) for field in FIELDS}
    if not isinstance(output["summary_vi"], str):
        raise ValueError("summary_vi must be a string")
    for field in FIELDS[1:]:
        if not isinstance(output[field], list):
            raise ValueError(f"{field} must be a list")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse downloaded Gemini Batch JSONL into validated scene records.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input or ROOT / "workspace" / "gemini_batches" / args.video_id / "responses.jsonl"
    output = args.output or ROOT / "workspace" / "gemini_parsed" / args.video_id / "scenes.jsonl"
    parsed, errors, seen = [], [], set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            envelope = json.loads(line)
            key = str(envelope.get("key") or "")
            if not key or key in seen:
                raise ValueError("missing or duplicate key")
            description = normalize(json.loads(response_text(envelope)))
            parsed.append({"scene_id": key, "video_id": args.video_id, "description": description,
                           "provenance": {"source": "9router_chat", "input": str(source), "line": line_number}})
            seen.add(key)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"line": line_number, "error": str(exc)})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in parsed), encoding="utf-8")
    temporary.replace(output)
    report = {"video_id": args.video_id, "input_lines": len(parsed) + len(errors), "parsed": len(parsed),
              "errors": errors, "output": str(output)}
    (output.parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
