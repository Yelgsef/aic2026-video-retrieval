from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def image_part(path: Path) -> dict:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
            "detail": "high",
        },
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Send one direct OpenAI-compatible request to 9Router and print the raw response."
    )
    parser.add_argument("--prompt", default="Reply with exactly: ROUTER_OK")
    parser.add_argument("--image", type=Path, help="Optional local JPG/PNG/WebP image")
    parser.add_argument("--url", default=os.getenv(
        "NINEROUTER_CHAT_URL", "http://localhost:20128/v1/chat/completions"))
    parser.add_argument("--model", default=os.getenv(
        "NINEROUTER_MODEL", "ag/gemini-3-flash"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json-object", action="store_true",
                        help="Ask the provider for JSON object output")
    args = parser.parse_args()

    content: list[dict] = [{"type": "text", "text": args.prompt}]
    if args.image:
        path = args.image.expanduser().resolve()
        if not path.is_file():
            parser.error(f"Image does not exist: {path}")
        content.append(image_part(path))

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.json_object:
        body["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "AIC2026-Pipeline/1.0",
    }
    key = os.getenv("NINEROUTER_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    request = Request(args.url, data=json.dumps(body).encode("utf-8"),
                      headers=headers, method="POST")
    print(json.dumps({
        "request": {
            "url": args.url,
            "model": args.model,
            "prompt": args.prompt,
            "image": str(args.image.resolve()) if args.image else None,
            "image_bytes": args.image.stat().st_size if args.image else 0,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "authorization_configured": bool(key),
        }
    }, ensure_ascii=False, indent=2), flush=True)

    try:
        with urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(json.dumps({"http_status": exc.code, "error": error_body},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    except URLError as exc:
        print(json.dumps({"network_error": str(exc.reason)}, ensure_ascii=False, indent=2),
              file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps({"response": payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
