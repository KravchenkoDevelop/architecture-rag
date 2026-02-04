from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

import requests


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")  # поставь свой
OLLAMA_TIMEOUT_SEC = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "120"))


class OllamaError(RuntimeError):
    pass


def _post(url: str, payload: Dict[str, Any]) -> requests.Response:
    return requests.post(
        url,
        json=payload,
        timeout=OLLAMA_TIMEOUT_SEC,
        headers={"Content-Type": "application/json"},
    )


def generate(system: str, user: str) -> str:
    """
    Tries (in order):
      1) Ollama native chat:     POST /api/chat
      2) Ollama native generate: POST /api/generate
      3) OpenAI-compatible:      POST /v1/chat/completions
    Returns plain text.
    Raises OllamaError with actionable info otherwise.
    """
    last_err: Optional[str] = None

    # 1) /api/chat (native)
    try:
        r = _post(
            f"{OLLAMA_BASE_URL}/api/chat",
            {
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": user or ""},
                ],
            },
        )
        if r.status_code == 404:
            last_err = "404 on /api/chat"
        else:
            r.raise_for_status()
            data = r.json()
            # typical: {"message": {"role":"assistant","content":"..."}, ...}
            msg = (data.get("message") or {}).get("content")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            last_err = f"unexpected /api/chat response shape: keys={list(data.keys())}"
    except Exception as e:
        last_err = f"/api/chat failed: {e!r}"

    # 2) /api/generate (native)
    try:
        r = _post(
            f"{OLLAMA_BASE_URL}/api/generate",
            {
                "model": OLLAMA_MODEL,
                "prompt": f"{system}\n\n{user}".strip(),
                "stream": False,
            },
        )
        if r.status_code == 404:
            last_err = (last_err or "") + " | 404 on /api/generate"
        else:
            r.raise_for_status()
            data = r.json()
            # typical: {"response":"...", ...}
            txt = data.get("response")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            last_err = f"unexpected /api/generate response shape: keys={list(data.keys())}"
    except Exception as e:
        last_err = (last_err or "") + f" | /api/generate failed: {e!r}"

    # 3) /v1/chat/completions (OpenAI-compatible)
    try:
        r = _post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": user or ""},
                ],
                "stream": False,
            },
        )
        if r.status_code == 404:
            last_err = (last_err or "") + " | 404 on /v1/chat/completions"
        else:
            r.raise_for_status()
            data = r.json()
            # typical OpenAI: {"choices":[{"message":{"content":"..."}}], ...}
            choices = data.get("choices") or []
            if choices and isinstance(choices, list):
                msg = (choices[0].get("message") or {}).get("content")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            last_err = f"unexpected /v1/chat/completions response shape: keys={list(data.keys())}"
    except Exception as e:
        last_err = (last_err or "") + f" | /v1/chat/completions failed: {e!r}"

    raise OllamaError(
        "Ollama request failed. "
        f"base_url={OLLAMA_BASE_URL!r} model={OLLAMA_MODEL!r} details={last_err}"
    )
