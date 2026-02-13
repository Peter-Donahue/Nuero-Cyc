from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json

import requests


class CycBridgeError(RuntimeError):
    """Raised when the CycBridgeServer returns an error (HTTP or application-level)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload
        self.text = text

    @property
    def server_message(self) -> str:
        """Best-effort extraction of a human-readable message from the server payload."""
        if isinstance(self.payload, dict):
            m = self.payload.get("message")
            if isinstance(m, str) and m.strip():
                return m.strip()
        return str(self)


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    session_mt: str
    genl_mt: str


class CycBridgeClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(text)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = requests.get(url, timeout=60)
        except requests.exceptions.RequestException as e:
            raise CycBridgeError(f"CycBridge request failed: {e}") from e

        text = r.text or ""
        payload = self._parse_json_object(text)

        if r.status_code != 200:
            msg = None
            if isinstance(payload, dict):
                msg = payload.get("message") or payload.get("error")
            raise CycBridgeError(
                f"CycBridge HTTP {r.status_code}: {msg or text[:500]}",
                status_code=r.status_code,
                payload=payload,
                text=text,
            )

        if payload is None:
            try:
                payload2 = r.json()
            except Exception as e:
                raise CycBridgeError(
                    f"CycBridge returned non-JSON response: {text[:500]}",
                    status_code=r.status_code,
                    payload=None,
                    text=text,
                ) from e
            if not isinstance(payload2, dict):
                raise CycBridgeError(
                    f"CycBridge returned unexpected JSON (not an object): {payload2!r}",
                    status_code=r.status_code,
                    payload=None,
                    text=text,
                )
            payload = payload2

        return payload

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=payload, timeout=300)
        except requests.exceptions.RequestException as e:
            raise CycBridgeError(f"CycBridge request failed: {e}") from e

        text = r.text or ""

        # The server often returns a JSON object even when HTTP != 200.
        resp_payload: Optional[Dict[str, Any]] = self._parse_json_object(text)
        if resp_payload is None:
            try:
                tmp = r.json()
                if isinstance(tmp, dict):
                    resp_payload = tmp
            except Exception:
                resp_payload = None

        if r.status_code != 200:
            msg = None
            if isinstance(resp_payload, dict):
                msg = resp_payload.get("message") or resp_payload.get("error")
            raise CycBridgeError(
                f"CycBridge HTTP {r.status_code}: {msg or text[:500]}",
                status_code=r.status_code,
                payload=resp_payload,
                text=text,
            )

        if resp_payload is None:
            raise CycBridgeError(
                f"CycBridge returned non-JSON response: {text[:500]}",
                status_code=r.status_code,
                payload=None,
                text=text,
            )

        if not resp_payload.get("ok", False):
            msg = resp_payload.get("message") or resp_payload.get("error") or str(resp_payload)
            raise CycBridgeError(
                f"CycBridge error: {msg}",
                status_code=r.status_code,
                payload=resp_payload,
                text=text,
            )

        return resp_payload

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def ensure_session(self, *, session_id: str, comment: str, genl_mt: str = "#$BaseKB") -> SessionInfo:
        data = self._post("/api/v1/session", {"session_id": session_id, "comment": comment, "genl_mt": genl_mt})
        return SessionInfo(session_id=data["session_id"], session_mt=data["session_mt"], genl_mt=data["genl_mt"])

    def constant_exists(self, name: str) -> bool:
        data = self._post("/api/v1/constant/exists", {"name": name})
        return bool(data.get("exists", False))

    def create_constant(self, name: str) -> str:
        data = self._post("/api/v1/constant/create", {"name": name})
        return str(data["name"])

    def assert_sentence(self, *, mt: str, sentence: str) -> None:
        self._post("/api/v1/assert", {"mt": mt, "sentence": sentence})

    def ask_true(self, *, mt: str, query: str) -> bool:
        data = self._post("/api/v1/ask_true", {"mt": mt, "query": query})
        return bool(data.get("answer", False))

    def ask_var(self, *, mt: str, query: str, var: str = "?X", limit: int = 50) -> List[str]:
        data = self._post("/api/v1/ask_var", {"mt": mt, "query": query, "var": var, "limit": limit})
        return [str(x) for x in data.get("bindings", [])]

    def converse(self, *, subl: str) -> str:
        data = self._post("/api/v1/converse", {"subl": subl})
        return str(data.get("result", ""))
