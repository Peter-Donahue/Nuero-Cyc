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
    # Cyc bridge server (Java wrapper around OpenCyc)
    cyc_bridge_base_url: str = _getenv("CYC_BRIDGE_BASE_URL", "http://localhost:8081")

    # Stanford CoreNLP server (used by corenlp_to_cycl.py)
    corenlp_base_url: str = _getenv("CORENLP_BASE_URL", "http://localhost:9000")

    # Shared HTTP timeout (seconds) for both CoreNLP and Cyc bridge calls.
    http_timeout_sec: int = _getenv_int("HTTP_TIMEOUT_SEC", 60)

    # Limit bindings returned by /ask_var.
    default_bindings_limit: int = _getenv_int("CYC_BINDINGS_LIMIT", 50)

    # Maximum number of missing-term repair loops for a single user prompt.
    max_missing_term_repairs: int = _getenv_int("MAX_MISSING_TERM_REPAIRS", 20)

    # The microtheory comment for session MTs.
    session_mt_comment: str = _getenv(
        "CYC_SESSION_MT_COMMENT",
        "Auto-created session microtheory for Cyc NL->CycL bridge.",
    )

    # Default general microtheory for session MTs.
    session_mt_genl: str = _getenv("CYC_SESSION_GENL_MT", "#$BaseKB")

    # Cyc lexicon + scoring configuration (for English -> Cyc constant mapping).
    cyc_lexicon_mt: str = _getenv("CYC_LEXICON_MT", "#$EnglishMt")
    cyc_query_mt: str = _getenv("CYC_QUERY_MT", "#$BaseKB")
    cyc_lex_limit: int = _getenv_int("CYC_LEX_LIMIT", 10)
    use_cyc_lexicon: bool = _getenv_bool("USE_CYC_LEXICON", True)
    use_cyc_scorer: bool = _getenv_bool("USE_CYC_SCORER", True)

    # Natural language rendering for result terms (best-effort via Cyc lexical predicates).
    use_cyc_nl: bool = _getenv_bool("USE_CYC_NL", True)
    cyc_nl_mt: str = _getenv("CYC_NL_MT", "")  # if empty, defaults to cyc_lexicon_mt

    # LLM-based KB augmentation via Ollama.
    # When a query returns no results, ask an LLM for factual assertions,
    # assert them into the session MT, then re-run the query.
    use_llm_kb_augmentation: bool = _getenv_bool("USE_LLM_KB_AUGMENTATION", True)
    ollama_base_url: str = _getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = _getenv("OLLAMA_MODEL", "llama3")
    ollama_timeout_sec: int = _getenv_int("OLLAMA_TIMEOUT_SEC", 120)

    # UI behavior.
    show_progress: bool = _getenv_bool("CYC_LLM_PROGRESS", True)
