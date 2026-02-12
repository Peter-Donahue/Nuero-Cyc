from __future__ import annotations

import os
from dataclasses import dataclass


def _getenv(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v else default


def _getenv_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _getenv_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _getenv_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    v = v.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


@dataclass(frozen=True)
class Settings:
    cyc_bridge_base_url: str = _getenv("CYC_BRIDGE_BASE_URL", "http://localhost:8081")
    ollama_base_url: str = _getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = _getenv("OLLAMA_MODEL", "llama3.2")
    ollama_temperature: float = _getenv_float("OLLAMA_TEMPERATURE", 0.0)

    # How many times to ask the LLM to repair a failing Cyc plan.
    max_plan_retries: int = _getenv_int("MAX_PLAN_RETRIES", 3)

    # Limit bindings returned by /ask_var.
    default_bindings_limit: int = _getenv_int("CYC_BINDINGS_LIMIT", 50)

    # The microtheory comment for session MTs.
    session_mt_comment: str = _getenv(
        "CYC_SESSION_MT_COMMENT",
        "Auto-created session microtheory for Cyc LLM bridge."
    )

    # Default general microtheory for session MTs.
    session_mt_genl: str = _getenv("CYC_SESSION_GENL_MT", "#$BaseKB")

    # Tracing (JSONL). CLI flags can override.
    llm_trace_dir: str = _getenv("LLM_TRACE_DIR", ".")

    # UI behavior.
    show_progress: bool = _getenv_bool("CYC_LLM_PROGRESS", True)
