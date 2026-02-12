from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class CycBridgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    session_mt: str
    genl_mt: str


class CycBridgeClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            raise CycBridgeError(f"CycBridge HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=300)
        if r.status_code != 200:
            raise CycBridgeError(f"CycBridge HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        if not data.get("ok", False):
            raise CycBridgeError(f"CycBridge error: {data}")
        return data

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def ensure_session(self, *, session_id: str, comment: str, genl_mt: str = "#$BaseKB") -> SessionInfo:
        data = self._post("/api/v1/session", {"session_id": session_id, "comment": comment, "genl_mt": genl_mt})
        return SessionInfo(session_id=data["session_id"], session_mt=data["session_mt"], genl_mt=data["genl_mt"])

    def constant_exists(self, name: str) -> bool:
        data = self._post("/api/v1/constant/exists", {"name": name})
        return bool(data["exists"])

    def create_constant(self, name: str) -> str:
        data = self._post("/api/v1/constant/create", {"name": name})
        return str(data["name"])

    def assert_sentence(self, *, mt: str, sentence: str) -> None:
        self._post("/api/v1/assert", {"mt": mt, "sentence": sentence})

    def ask_true(self, *, mt: str, query: str) -> bool:
        data = self._post("/api/v1/ask_true", {"mt": mt, "query": query})
        return bool(data["answer"])

    def ask_var(self, *, mt: str, query: str, var: str = "?X", limit: int = 50) -> List[str]:
        data = self._post("/api/v1/ask_var", {"mt": mt, "query": query, "var": var, "limit": limit})
        return [str(x) for x in data.get("bindings", [])]

    def converse(self, *, subl: str) -> str:
        data = self._post("/api/v1/converse", {"subl": subl})
        return str(data.get("result", ""))
