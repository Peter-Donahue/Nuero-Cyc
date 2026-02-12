from __future__ import annotations

import json
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .config import Settings
from .orchestrator import Orchestrator, RunResult


HELP_TEXT = """Commands:
  /help                 Show this help
  /debug on|off         Toggle debug output
  /trace on|off [path]  Enable/disable LLM call tracing (writes JSONL to a file)
  /session              Show current session microtheory
  /health               Check Cyc bridge health
  /exit                 Quit
"""


def run_cli(
    *,
    settings: Settings,
    debug: bool,
    allow_converse: bool,
    once: Optional[str],
    llm_trace: Optional[str] = None,
    show_progress: bool = True,
) -> int:
    console = Console()

    orch = Orchestrator(settings)
    orch.set_debug(debug)
    orch.allow_converse = allow_converse

    # Progress messages
    if show_progress and settings.show_progress:
        orch.set_progress_callback(lambda m: console.print(f"[dim]{m}[/dim]"))
    else:
        orch.set_progress_callback(None)

    # Optional LLM trace
    trace_path: Optional[str] = None
    if llm_trace:
        trace_path = orch.enable_trace(llm_trace)

    # Health check
    try:
        health = orch.cyc.health()
        status_lines = [
            "[bold]Cyc Bridge[/bold]",
            f"base_url: {settings.cyc_bridge_base_url}",
            f"cyc_connected: {health.get('cyc_connected')}",
            f"cyc_host: {health.get('cyc_host')}  cyc_port: {health.get('cyc_port')}",
            f"session_mt: {orch.session_info.session_mt}",
            f"ollama: {settings.ollama_base_url}  model: {settings.ollama_model}",
        ]
        if trace_path:
            status_lines.append(f"llm_trace: {trace_path}")
        console.print(Panel.fit("\n".join(status_lines), title="Status"))
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Startup error[/bold red]\n{e}", title="Error"))
        orch.close()
        return 2

    if once:
        try:
            return _run_once(console, orch, once)
        finally:
            orch.close()

    console.print(Panel.fit(HELP_TEXT.strip(), title="Help"))

    try:
        while True:
            try:
                line = Prompt.ask("cyc-llm").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not line:
                continue

            if line.startswith("/"):
                if _handle_command(console, orch, line):
                    break
                continue

            try:
                res = orch.handle_user_prompt(line)
            except Exception as e:
                console.print(Panel.fit(f"[bold red]Error[/bold red]\n{e}", title="Run failed"))
                continue

            _render_result(console, res, debug=orch.ctx.debug)

        return 0
    finally:
        orch.close()


def _run_once(console: Console, orch: Orchestrator, prompt: str) -> int:
    try:
        res = orch.handle_user_prompt(prompt)
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Error[/bold red]\n{e}", title="Run failed"))
        return 2
    _render_result(console, res, debug=orch.ctx.debug)
    return 0


def _handle_command(console: Console, orch: Orchestrator, line: str) -> bool:
    parts = line.strip().split()
    cmd = parts[0].lower()

    if cmd in ("/exit", "/quit"):
        return True

    if cmd == "/help":
        console.print(Panel.fit(HELP_TEXT.strip(), title="Help"))
        return False

    if cmd == "/debug":
        if len(parts) < 2:
            console.print(f"debug is {'on' if orch.ctx.debug else 'off'}")
            return False
        v = parts[1].lower()
        orch.set_debug(v in ("1", "true", "on", "yes"))
        console.print(f"debug set to {'on' if orch.ctx.debug else 'off'}")
        return False

    if cmd == "/trace":
        if len(parts) < 2:
            if orch.trace_path:
                console.print(f"llm trace is on: {orch.trace_path}")
            else:
                console.print("llm trace is off")
            return False

        arg = parts[1]
        arg_l = arg.lower()
        if arg_l in ("1", "true", "on", "yes"):
            path = orch.enable_trace("auto")
            console.print(f"llm trace enabled: {path}")
            return False
        if arg_l in ("0", "false", "off", "no"):
            orch.disable_trace()
            console.print("llm trace disabled")
            return False

        # treat as path
        path = orch.enable_trace(arg)
        console.print(f"llm trace enabled: {path}")
        return False

    if cmd == "/session":
        console.print(Panel.fit(
            f"session_id: {orch.session_info.session_id}\n"
            f"session_mt: {orch.session_info.session_mt}\n"
            f"genl_mt: {orch.session_info.genl_mt}\n",
            title="Session"
        ))
        return False

    if cmd == "/health":
        try:
            health = orch.cyc.health()
            console.print(Panel.fit(json.dumps(health, indent=2), title="Health"))
        except Exception as e:
            console.print(Panel.fit(f"[bold red]Error[/bold red]\n{e}", title="Health failed"))
        return False

    console.print(f"Unknown command: {cmd}. Try /help.")
    return False


def _render_result(console: Console, res: RunResult, debug: bool) -> None:
    console.print(Panel(res.answer.strip() or "(empty)", title="Answer"))

    if res.cyc_evidence:
        console.print(Panel("\n".join(f"- {x}" for x in res.cyc_evidence), title="Cyc evidence"))

    if res.limitations:
        console.print(Panel("\n".join(f"- {x}" for x in res.limitations), title="Limitations"))

    if debug:
        console.print(Panel(json.dumps(res.debug, indent=2)[:15000], title="Debug (truncated)"))
