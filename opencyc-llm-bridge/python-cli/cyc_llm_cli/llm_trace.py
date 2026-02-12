from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for LLM call tracing."""

    path: str


class LLMTrace:
    """Write a detailed, append-only JSONL trace to disk.

    Each call to :meth:`write` appends a single JSON object on one line.
    """

    def __init__(self, cfg: TraceConfig):
        self.cfg = cfg
        self.path = cfg.path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        rec: Dict[str, Any] = {"event": event, "ts": time.time()}
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._fp.write(line + "\n")
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fp.flush()
            finally:
                self._fp.close()


class NullTrace(LLMTrace):
    """A drop-in no-op trace implementation."""

    def __init__(self) -> None:  # type: ignore[override]
        # Do not call super().__init__.
        self.cfg = TraceConfig(path="")
        self.path = ""

    def write(self, event: str, **fields: Any) -> None:  # type: ignore[override]
        return

    def close(self) -> None:  # type: ignore[override]
        return
