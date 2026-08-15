from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aic_pipeline.router_chat import RouterChatClient


class OcrLine(BaseModel):
    text: str
    frame_label: str


class SceneResult(BaseModel):
    summary_vi: str
    ocr_lines: list[OcrLine]
    people: list[str]
    objects: list[str]
    actions: list[str]
    locations: list[str]
    events: list[str]
    search_keywords_vi: list[str]
    uncertainties: list[str]


class AsrResult(BaseModel):
    normalized_text: str
    proper_nouns: list[str]
    changes: list[str]
    uncertainties: list[str]


class OcrResult(BaseModel):
    text: str
    readable: bool
    language: str
    uncertainty: str


def paths(task: str, video_id: str) -> tuple[Path, Path, type[BaseModel]]:
    if task == "scenes":
        root = ROOT / "workspace/gemini_batches" / video_id
        return root / "requests.jsonl", root / "responses.jsonl", SceneResult
    if task == "asr-normalize":
        root = ROOT / "workspace/asr_normalization" / video_id
        return root / "requests.jsonl", root / "responses.jsonl", AsrResult
    root = ROOT / "workspace/ocr_fullres" / video_id
    return root / "gemini_requests.jsonl", root / "responses.jsonl", OcrResult


def unpack_request(row: dict[str, Any]) -> tuple[str, Image.Image | None]:
    parts = row["request"]["contents"][0]["parts"]
    prompt = "\n".join(str(part["text"]) for part in parts if "text" in part)
    image = None
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            image = Image.open(BytesIO(base64.b64decode(inline["data"]))).convert("RGB")
            break
    return prompt, image


def legacy_envelope(key: str, text: str, usage: dict[str, Any] | None) -> dict[str, Any]:
    return {"key": key, "response": {"candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": usage or {}, "provider": "9router"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable parallel 9Router enrichment.")
    parser.add_argument("--task", choices=("scenes", "asr-normalize", "ocr-crops"), required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--model")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source, output, schema = paths(args.task, args.video_id)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    done: dict[str, dict[str, Any]] = {}
    if output.exists() and not args.force:
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                provider = item.get("provider") or (item.get("response") or {}).get("provider")
                if item.get("key") and not item.get("error") and provider == "9router":
                    done[str(item["key"])] = item
    client = RouterChatClient(ROOT, timeout_seconds=180)

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        prompt, image = unpack_request(row)
        error = ""
        for attempt in range(args.retries + 1):
            try:
                token_limit = 4096 if args.task == "scenes" else 2048
                response = client.json_response(prompt, schema, image, model=args.model,
                                                temperature=0.1, max_tokens=token_limit)
                return legacy_envelope(str(row["key"]), response.text, response.usage)
            except Exception as exc:
                error = str(exc)
                if attempt < args.retries:
                    time.sleep(min(30, 2 ** attempt))
        # Remote normalization/OCR is supplemental. Preserve a valid, explicit
        # local-evidence fallback instead of aborting an entire dataset shard.
        if args.task == "asr-normalize":
            marker = "Đoạn cần sửa:"
            raw = prompt.split(marker, 1)[1].split("\n", 1)[0].strip() if marker in prompt else ""
            value = AsrResult(normalized_text=raw, proper_nouns=[], changes=[],
                              uncertainties=["9Router output invalid; kept raw ASR"])
        elif args.task == "ocr-crops":
            value = OcrResult(text="", readable=False, language="unknown",
                              uncertainty="9Router output invalid; local OCR retained")
        else:
            value = SceneResult(summary_vi="", ocr_lines=[], people=[], objects=[], actions=[],
                                locations=[], events=[], search_keywords_vi=[],
                                uncertainties=["9Router output invalid; local evidence retained"])
        fallback = legacy_envelope(str(row["key"]), value.model_dump_json(), None)
        fallback["fallback"] = error
        return fallback

    pending = [row for row in rows if str(row["key"]) not in done]
    completed = dict(done)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(execute, row): str(row["key"]) for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed[str(result["key"])] = result
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(json.dumps(completed[str(row["key"])], ensure_ascii=False) + "\n"
                                      for row in rows if str(row["key"]) in completed), encoding="utf-8")
            print(f"9ROUTER_PROGRESS {args.task} {args.video_id} {len(done)+index}/{len(rows)}", flush=True)
    failed = sum(bool(item.get("error")) for item in completed.values())
    print(json.dumps({"task": args.task, "video_id": args.video_id, "requests": len(rows),
                      "completed": len(completed) - failed, "failed": failed,
                      "output": str(output)}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
