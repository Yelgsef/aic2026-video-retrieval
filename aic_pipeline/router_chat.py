from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Type
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel


@dataclass
class RouterResponse:
    text: str
    usage: dict[str, Any] | None
    raw: dict[str, Any]


class RouterChatClient:
    """Minimal OpenAI-compatible client for the configured 9Router gateway."""

    def __init__(self, root: Path, timeout_seconds: float = 30.0) -> None:
        load_dotenv(root.resolve() / ".env")
        self.url = os.getenv(
            "NINEROUTER_CHAT_URL", "http://localhost:20128/v1/chat/completions"
        ).strip()
        self.key = os.getenv("NINEROUTER_KEY", "").strip()
        self.model = os.getenv("NINEROUTER_MODEL", "gc/gemini-2.5-flash").strip()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _image_part(image: Image.Image) -> dict[str, Any]:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}}

    def json_response(
        self,
        prompt: str,
        schema: Type[BaseModel],
        image: Image.Image | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> RouterResponse:
        schema_json = schema.model_json_schema()
        instruction = (
            prompt + "\nReturn only one JSON object matching this JSON Schema:\n"
            + json.dumps(schema_json, ensure_ascii=False)
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        if image is not None:
            content.append(self._image_part(image))
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AIC2026-Pipeline/1.0",
        }
        # 9Router docs say Authorization must be omitted when gateway auth is disabled.
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"9Router HTTP {exc.code}: {body.strip() or exc.reason}. "
                "If remote API auth is enabled, set NINEROUTER_KEY in .env."
            ) from exc
        text = str(raw["choices"][0]["message"]["content"]).strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text,
                          re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        start = text.find("{")
        if start >= 0:
            # Some gateways append a second JSON object or prose despite
            # response_format. Decode only the first complete object.
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            text = json.dumps(value, ensure_ascii=False)
        # Validate at the boundary so malformed router/provider output never
        # reaches retrieval or QA state.
        validated = schema.model_validate_json(text)
        return RouterResponse(validated.model_dump_json(), raw.get("usage"), raw)


class GeminiApiClient(RouterChatClient):
    """Direct Google Gemini API client with the same structured-output interface."""

    def __init__(self, root: Path, timeout_seconds: float = 30.0) -> None:
        load_dotenv(root.resolve() / ".env")
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_API_MODEL", "gemini-2.5-flash").strip()
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for gemini_api mode")

    def json_response(
        self,
        prompt: str,
        schema: Type[BaseModel],
        image: Image.Image | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> RouterResponse:
        schema_json = schema.model_json_schema()
        parts: list[dict[str, Any]] = [{"text": prompt + "\nReturn JSON only."}]
        if image is not None:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                           "data": base64.b64encode(buffer.getvalue()).decode("ascii")}})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseSchema": schema_json,
            },
        }
        target_model = model or self.model
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{target_model}:generateContent?key={self.api_key}")
        request = Request(url, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {body.strip() or exc.reason}") from exc
        try:
            text = str(raw["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gemini API returned no text: {raw}") from exc
        start = text.find("{")
        if start >= 0:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            text = json.dumps(value, ensure_ascii=False)
        validated = schema.model_validate_json(text)
        return RouterResponse(validated.model_dump_json(), raw.get("usageMetadata"), raw)
