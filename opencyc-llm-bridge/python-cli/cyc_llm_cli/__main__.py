from __future__ import annotations

import argparse
import os

from .config import Settings
from .cli import run_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenCyc ⇄ CoreNLP CLI bridge (English → CycL)")
    parser.add_argument(
        "--bridge",
        help="Cyc bridge base URL, e.g. http://localhost:8081 (overrides CYC_BRIDGE_BASE_URL env)",
    )
    parser.add_argument(
        "--corenlp",
        help="CoreNLP base URL, e.g. http://localhost:9000 (overrides CORENLP_BASE_URL env)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages in the CLI.",
    )
    parser.add_argument("--once", help="Run one question and exit")

    # Lexicon/scoring toggles
    parser.add_argument("--no-cyc-lex", action="store_true", help="Disable Cyc lexicon lookups")
    parser.add_argument("--no-cyc-score", action="store_true", help="Disable Cyc candidate scoring")
    parser.add_argument("--no-cyc-nl", action="store_true", help="Disable Cyc term→English rendering for results")
    parser.add_argument(
        "--lex-mt",
        help="Microtheory for Cyc lexicon queries (overrides CYC_LEXICON_MT env)",
    )
    parser.add_argument(
        "--query-mt",
        help="Microtheory for Cyc scoring/type queries (overrides CYC_QUERY_MT env)",
    )
    parser.add_argument(
        "--lex-limit",
        type=int,
        help="Max results per Cyc lexicon lookup (overrides CYC_LEX_LIMIT env)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="HTTP timeout seconds for CoreNLP + Cyc bridge (overrides HTTP_TIMEOUT_SEC env)",
    )

    args = parser.parse_args(argv)

    # Env overrides (simple)
    if args.bridge:
        os.environ["CYC_BRIDGE_BASE_URL"] = args.bridge
    if args.corenlp:
        os.environ["CORENLP_BASE_URL"] = args.corenlp
    if args.lex_mt:
        os.environ["CYC_LEXICON_MT"] = args.lex_mt
    if args.query_mt:
        os.environ["CYC_QUERY_MT"] = args.query_mt
    if args.lex_limit is not None:
        os.environ["CYC_LEX_LIMIT"] = str(int(args.lex_limit))
    if args.timeout is not None:
        os.environ["HTTP_TIMEOUT_SEC"] = str(int(args.timeout))

    if args.no_cyc_lex:
        os.environ["USE_CYC_LEXICON"] = "0"
    if args.no_cyc_score:
        os.environ["USE_CYC_SCORER"] = "0"
    if args.no_cyc_nl:
        os.environ["USE_CYC_NL"] = "0"

    settings = Settings()
    return run_cli(
        settings=settings,
        debug=args.debug,
        once=args.once,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    raise SystemExit(main())
