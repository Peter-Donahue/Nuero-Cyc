from __future__ import annotations

import argparse
import os
import sys

from .config import Settings
from .cli import run_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenCyc ⇄ Local LLM CLI bridge (Ollama by default)")
    parser.add_argument("--model", help="Ollama model name (overrides OLLAMA_MODEL env)")
    parser.add_argument("--ollama", help="Ollama base URL, e.g. http://localhost:11434 (overrides OLLAMA_BASE_URL env)")
    parser.add_argument("--bridge", help="Cyc bridge base URL, e.g. http://localhost:8081 (overrides CYC_BRIDGE_BASE_URL env)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--llm-trace",
        nargs="?",
        const="auto",
        help="Write a detailed JSONL trace of all LLM calls (optional file path).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages in the CLI.",
    )
    parser.add_argument("--allow-converse", action="store_true", help="Allow raw SubL via /api/v1/converse")
    parser.add_argument("--once", help="Run one question and exit")
    args = parser.parse_args(argv)

    # Env overrides (simple)
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    if args.ollama:
        os.environ["OLLAMA_BASE_URL"] = args.ollama
    if args.bridge:
        os.environ["CYC_BRIDGE_BASE_URL"] = args.bridge

    settings = Settings()
    return run_cli(
        settings=settings,
        debug=args.debug,
        allow_converse=args.allow_converse,
        once=args.once,
        llm_trace=args.llm_trace,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    raise SystemExit(main())
