from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """
    Minimal Ollama REST client for /api/chat.

    Docs:
    - Structured outputs: pass JSON schema in `format` and use stream=false.
    """
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[Any] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Non-streaming call to /api/chat.

        For streaming token deltas, use :meth:`chat_stream_text`.
        """

        if stream:
            raise OllamaError("chat(stream=True) is not supported; use chat_stream_text().")

        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if format is not None:
            payload["format"] = format
        if options is not None:
            payload["options"] = options

        r = requests.post(url, json=payload, timeout=300)
        if r.status_code != 200:
            raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def chat_stream_text(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        on_chunk: Optional[Callable[[Dict[str, Any], int], None]] = None,
    ) -> str:
        """Streaming call to /api/chat that returns the concatenated assistant content.

        Ollama streams newline-delimited JSON objects. We parse each chunk, optionally invoking
        `on_chunk(chunk, i)`, and concatenate `chunk.message.content` as the final text.
        """

        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if format is not None:
            payload["format"] = format
        if options is not None:
            payload["options"] = options

        r = requests.post(url, json=payload, timeout=300, stream=True)
        if r.status_code != 200:
            raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")

        parts: List[str] = []
        i = 0
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as e:
                raise OllamaError(f"Invalid JSON chunk from Ollama stream: {line[:200]}") from e

            if on_chunk is not None:
                on_chunk(chunk, i)

            msg = chunk.get("message", {}) or {}
            delta = msg.get("content", "")
            if isinstance(delta, str) and delta:
                parts.append(delta)

            i += 1
            if chunk.get("done") is True:
                break

        return "".join(parts)

    def chat_text(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Dict[str, Any], int], None]] = None,
    ) -> str:
        if stream:
            return self.chat_stream_text(model=model, messages=messages, format=format, options=options, on_chunk=on_chunk)

        data = self.chat(model=model, messages=messages, format=format, stream=False, options=options)
        if on_chunk is not None:
            on_chunk(data, 0)
        msg = data.get("message", {}) or {}
        content = msg.get("content", "")
        if not isinstance(content, str):
            return json.dumps(content)
        return content

    def chat_json(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Dict[str, Any], int], None]] = None,
    ) -> Any:
        # In practice, also include schema in prompt for grounding (per Ollama docs),
        # but the caller should do that; this method only enforces `format`.
        raw = self.chat_text(model=model, messages=messages, format=schema, options=options, stream=stream, on_chunk=on_chunk)
        return _safe_json_loads(raw)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _safe_json_loads(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        raise OllamaError("Empty response when JSON was expected.")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to extract a JSON object from text (models sometimes add prose).
        m = _JSON_RE.search(s)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise OllamaError(f"Model did not return valid JSON. Raw: {s[:500]}")
