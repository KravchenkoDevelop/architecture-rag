from __future__ import annotations

import os
import json
import requests
from typing import Optional


class OllamaError(RuntimeError):
    pass


def _norm_base_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return "http://ollama:11434"
    url = url.rstrip("/")

    # убираем частые ошибочные хвосты
    for suffix in ("/api", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            url = url.rstrip("/")
    return url


OLLAMA_BASE_URL = _norm_base_url(os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SEC", "120"))


def _post(path: str, payload: dict) -> requests.Response:
    url = f"{OLLAMA_BASE_URL}{path}"
    return requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)


def generate(system: str, user: str) -> str:
    details = []

    # 1) /api/chat
    try:
        r = _post(
            "/api/chat",
            {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
        )
        if r.status_code == 200:
            data = r.json()
            return (data.get("message") or {}).get("content", "") or ""
        details.append(f"{r.status_code} on /api/chat: {r.text[:200]}")
    except Exception as e:
        details.append(f"exc on /api/chat: {type(e).__name__}")

    # 2) /api/generate
    try:
        r = _post(
            "/api/generate",
            {
                "model": OLLAMA_MODEL,
                "prompt": f"{system}\n\n{user}",
                "stream": False,
            },
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("response", "") or ""
        details.append(f"{r.status_code} on /api/generate: {r.text[:200]}")
    except Exception as e:
        details.append(f"exc on /api/generate: {type(e).__name__}")

    # 3) OpenAI-совместимый /v1/chat/completions (если включен)
    try:
        r = _post(
            "/v1/chat/completions",
            {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        if r.status_code == 200:
            data = r.json()
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                return msg.get("content", "") or ""
        details.append(f"{r.status_code} on /v1/chat/completions: {r.text[:200]}")
    except Exception as e:
        details.append(f"exc on /v1/chat/completions: {type(e).__name__}")

    raise OllamaError(
        f"Ollama request failed. base_url={OLLAMA_BASE_URL!r} model={OLLAMA_MODEL!r} details={' | '.join(details)}"
    )
