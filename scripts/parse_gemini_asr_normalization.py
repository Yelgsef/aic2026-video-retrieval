from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def response_text(envelope: dict) -> str:
    response = envelope.get("response", envelope)
    candidates = response.get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    return "".join(str(x.get("text", "")) for x in parts).removeprefix("```json").removesuffix("```").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Gemini ASR corrections without losing raw transcript.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--responses", type=Path)
    args = parser.parse_args()
    root = ROOT / "workspace/asr_normalization" / args.video_id
    responses = args.responses or root / "responses.jsonl"
    corrections, errors = {}, []
    for number, line in enumerate(responses.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            envelope = json.loads(line); key = str(envelope["key"])
            value = json.loads(response_text(envelope))
            if not isinstance(value.get("normalized_text"), str):
                raise ValueError("missing normalized_text")
            corrections[key] = value
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"line": number, "error": str(exc)})
    source = json.loads((ROOT / "workspace/asr/faster_whisper_large_v3" /
                         f"{args.video_id}.json").read_text(encoding="utf-8"))
    normalized = []
    for segment in source["segments"]:
        key = f"{args.video_id}:segment:{int(segment['segment_id'])}"
        correction = corrections.get(key)
        normalized.append({"segment_id": int(segment["segment_id"]), "start": segment["start"],
                           "end": segment["end"], "raw_text": segment["text"],
                           "normalized_text": correction["normalized_text"] if correction else segment["text"],
                           "gemini_corrected": correction is not None,
                           "proper_nouns": correction.get("proper_nouns", []) if correction else [],
                           "changes": correction.get("changes", []) if correction else [],
                           "uncertainties": correction.get("uncertainties", []) if correction else []})
    output = root / "normalized.json"
    output.write_text(json.dumps({"schema_version": "0.1.0", "video_id": args.video_id,
        "segments": normalized, "full_text_raw": source.get("full_text", ""),
        "full_text_normalized": " ".join(x["normalized_text"].strip() for x in normalized),
        "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"video_id": args.video_id, "corrected": len(corrections),
                      "errors": len(errors), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
