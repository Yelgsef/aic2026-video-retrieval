from __future__ import annotations

import os
import re

import httpx
from dotenv import load_dotenv

from backend.config import settings


DATA_URL = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


async def answer_with_images(provider: str, api_key: str, model: str,
                             system_prompt: str, question: str,
                             images: list[dict]) -> str:
    load_dotenv(settings.root / ".env")
    if provider == "9router":
        content: list[dict] = [{"type": "text", "text": question.strip()}]
        for image in images:
            if not DATA_URL.fullmatch(image["data_url"]):
                raise ValueError("Invalid image data URL")
            content.append({"type": "image_url", "image_url": {"url": image["data_url"]}})
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})
        key = api_key.strip() or os.environ.get("NINEROUTER_KEY", "").strip()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": model.strip(), "messages": messages,
                   "temperature": 0.1, "max_tokens": 512, "stream": False}
        url = os.environ.get("NINEROUTER_CHAT_URL", "http://localhost:20128/v1/chat/completions").strip()
        async with httpx.AsyncClient(timeout=65) as client:
            response = await client.post(url, headers=headers, json=payload)
        result = response.json()
        if not response.is_success:
            raise RuntimeError(result.get("error", {}).get("message", f"9Router returned HTTP {response.status_code}"))
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(answer, list):
            answer = "".join(item.get("text", "") for item in answer if isinstance(item, dict))
        answer = str(answer).strip()
        if not answer:
            raise RuntimeError("9Router returned an empty answer")
        return answer
    if provider != "ai_studio":
        raise ValueError(f"Unsupported QA provider: {provider}")
    if not api_key.strip():
        raise ValueError("AI Studio API key is required")
    parts: list[dict] = [{"text": question.strip()}]
    for image in images:
        match = DATA_URL.fullmatch(image["data_url"])
        if not match:
            raise ValueError("Invalid image data URL")
        mime_type, encoded = match.groups()
        parts.append({"inlineData": {"mimeType": mime_type, "data": encoded}})
    body: dict = {"contents": [{"role": "user", "parts": parts}]}
    if system_prompt.strip():
        body["systemInstruction"] = {"role": "system", "parts": [{"text": system_prompt.strip()}]}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key.strip()}, json=body,
        )
    payload = response.json()
    if not response.is_success:
        message = payload.get("error", {}).get("message", f"AI Studio returned HTTP {response.status_code}")
        raise RuntimeError(message)
    candidates = payload.get("candidates", [])
    answer = "".join(part.get("text", "") for part in
                     candidates[0].get("content", {}).get("parts", []) if isinstance(part, dict)).strip()
    if not answer:
        raise RuntimeError("AI Studio returned an empty answer")
    return answer
