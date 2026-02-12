from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import Settings
from .cyc_bridge_client import CycBridgeClient, CycBridgeError, SessionInfo
from .llm_trace import LLMTrace, NullTrace, TraceConfig
from .ollama_client import OllamaClient, OllamaError
from .planner import (
    PlannerContext,
    answer_json_schema,
    build_answer_messages,
    build_planner_messages,
    plan_json_schema,
)


class PlanValidationError(RuntimeError):
    pass


class PlanExecutionError(RuntimeError):
    pass


class NoAnswerError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnswerSignal:
    action_index: int
    action_type: str  # ask_var | ask_true
    answer_text: str
    raw_value: Any
    query: Optional[str] = None


@dataclass
class RunResult:
    answer: str
    cyc_evidence: List[str]
    limitations: List[str]
    debug: Dict[str, Any]


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_jsonish(raw: str) -> Any:
    raw = (raw or "").strip()
    if not raw:
        raise OllamaError("Empty response when JSON was expected.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise OllamaError(f"Model did not return valid JSON. Raw: {raw[:500]}")


def _is_single_toplevel_cycl_form(s: str) -> bool:
    """Return True if `s` looks like a single, fully-parenthesized CycL expression.

    This is a defensive check to prevent common LLM failure modes like:
    - missing parentheses ("#$isa ...")
    - multiple top-level forms ("(?x) (...)")
    - variable-first forms
    """

    s = (s or "").strip()
    if not s:
        return False
    if not (s.startswith("(") and s.endswith(")")):
        return False

    # The head symbol in CycL should virtually always be a Cyc constant (e.g., #$isa, #$and).
    i = 1
    while i < len(s) and s[i].isspace():
        i += 1
    if s[i : i + 2] != "#$":
        return False

    depth = 0
    in_str = False
    esc = False

    for idx, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
            # If we close the outermost form before the end, ensure the rest is whitespace.
            if depth == 0 and idx != len(s) - 1:
                if any(not c.isspace() for c in s[idx + 1 :]):
                    return False

    return depth == 0 and not in_str


def _normalize_constant_ref(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return name
    return name if name.startswith("#$") else "#$" + name


class Orchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cyc = CycBridgeClient(settings.cyc_bridge_base_url)
        self.ollama = OllamaClient(settings.ollama_base_url)

        self.session_info = self._init_session()
        self.ctx = PlannerContext(
            session_mt=self.session_info.session_mt,
            session_genl_mt=self.session_info.genl_mt,
            debug=False,
        )
        self.allow_converse = False

        # conversation = [{"role":"user"/"assistant","content":"..."}]
        self.conversation: List[Dict[str, str]] = []

        # Optional progress callback (UI layer)
        self._progress_cb: Optional[Callable[[str], None]] = None

        # Optional LLM trace
        self.trace: Optional[LLMTrace] = None
        self.trace_path: Optional[str] = None

    # -----------------------------
    # Lifecycle
    # -----------------------------

    def close(self) -> None:
        self.disable_trace()

    def set_progress_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._progress_cb = cb

    def _progress(self, msg: str) -> None:
        if self._progress_cb is not None:
            self._progress_cb(msg)
        if self.trace is not None:
            self.trace.write("progress", message=msg)

    def enable_trace(self, path: Optional[str] = None) -> str:
        """Enable JSONL tracing of LLM calls (and related orchestration events)."""

        if path is None or path == "auto":
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            fname = f"llm_calls_{self.session_info.session_id}_{ts}.jsonl"
            path = os.path.join(self.settings.llm_trace_dir, fname)

        # Close existing trace if any
        self.disable_trace()

        t = LLMTrace(TraceConfig(path=path))
        self.trace = t
        self.trace_path = path

        t.write(
            "trace_start",
            session_id=self.session_info.session_id,
            session_mt=self.session_info.session_mt,
            genl_mt=self.session_info.genl_mt,
            cyc_bridge_base_url=self.settings.cyc_bridge_base_url,
            ollama_base_url=self.settings.ollama_base_url,
            ollama_model=self.settings.ollama_model,
        )

        return path

    def disable_trace(self) -> None:
        if self.trace is not None:
            try:
                self.trace.write("trace_end")
            except Exception:
                pass
            try:
                self.trace.close()
            finally:
                self.trace = None
                self.trace_path = None

    # -----------------------------
    # Session
    # -----------------------------

    def _init_session(self) -> SessionInfo:
        session_id = uuid.uuid4().hex[:12]
        return self.cyc.ensure_session(
            session_id=session_id,
            comment=self.settings.session_mt_comment,
            genl_mt=self.settings.session_mt_genl,
        )

    def set_debug(self, enabled: bool) -> None:
        self.ctx = PlannerContext(
            session_mt=self.ctx.session_mt,
            session_genl_mt=self.ctx.session_genl_mt,
            debug=enabled,
        )

    # -----------------------------
    # Public API
    # -----------------------------

    def handle_user_prompt(self, user_prompt: str) -> RunResult:
        self.conversation.append({"role": "user", "content": user_prompt})

        self._progress("interpreting prompt...")
        plan, results, exec_log, answer_signal = self._plan_and_execute_with_repairs(user_prompt)

        self._progress("formatting answer...")
        answer_json = self._generate_answer(user_prompt, results, exec_log, answer_signal)

        # Store assistant answer in conversation
        self.conversation.append({"role": "assistant", "content": answer_json["answer"]})

        debug_blob = {
            "session": {
                "session_id": self.session_info.session_id,
                "session_mt": self.session_info.session_mt,
                "genl_mt": self.session_info.genl_mt,
            },
            "plan": plan,
            "execution_log": exec_log,
            "cyc_results": results,
            "answer_signal": {
                "action_index": answer_signal.action_index,
                "action_type": answer_signal.action_type,
                "answer_text": answer_signal.answer_text,
                "query": answer_signal.query,
                "raw_value": answer_signal.raw_value,
            },
            "raw_answer_json": answer_json,
        }

        return RunResult(
            answer=answer_json["answer"],
            cyc_evidence=answer_json.get("cyc_evidence", []),
            limitations=answer_json.get("limitations", []),
            debug=debug_blob,
        )

    # -----------------------------
    # LLM calls (with tracing)
    # -----------------------------

    def _llm_chat_json(
        self,
        *,
        kind: str,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        options = options or {"temperature": self.settings.ollama_temperature}

        trace = self.trace
        call_id = uuid.uuid4().hex
        stream = trace is not None

        if trace is not None:
            trace.write(
                "llm_call_start",
                call_id=call_id,
                kind=kind,
                model=self.settings.ollama_model,
                stream=stream,
                options=options,
                messages=messages,
                schema=schema,
            )

        delta_counter = {"i": 0}

        def on_chunk(chunk: Dict[str, Any], i: int) -> None:
            if trace is None:
                return
            # log the raw chunk (includes done/usage timings)
            trace.write("llm_call_chunk", call_id=call_id, i=i, chunk=chunk)
            msg = chunk.get("message", {}) or {}
            delta = msg.get("content", "")
            if isinstance(delta, str) and delta:
                trace.write(
                    "llm_call_delta",
                    call_id=call_id,
                    i=delta_counter["i"],
                    delta=delta,
                    done=bool(chunk.get("done", False)),
                )
                delta_counter["i"] += 1

        try:
            raw = self.ollama.chat_text(
                model=self.settings.ollama_model,
                messages=messages,
                format=schema,
                options=options,
                stream=stream,
                on_chunk=on_chunk,
            )
        except Exception as e:
            if trace is not None:
                trace.write(
                    "llm_call_error",
                    call_id=call_id,
                    kind=kind,
                    error_type=e.__class__.__name__,
                    error=str(e),
                )
            raise

        if trace is not None:
            trace.write("llm_call_complete", call_id=call_id, kind=kind, raw=raw)

        parsed = None
        try:
            parsed = _parse_jsonish(raw)
        except Exception as e:
            if trace is not None:
                trace.write(
                    "llm_call_parse_error",
                    call_id=call_id,
                    kind=kind,
                    error_type=e.__class__.__name__,
                    error=str(e),
                    raw=raw[:2000],
                )
            raise

        if trace is not None:
            trace.write("llm_call_parsed", call_id=call_id, kind=kind, parsed=parsed)

        return parsed

    # -----------------------------
    # Planning + execution
    # -----------------------------

    def _plan_and_execute_with_repairs(
        self, user_prompt: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], AnswerSignal]:
        error_context: Optional[Dict[str, Any]] = None
        last_exception: Optional[Exception] = None

        for attempt in range(self.settings.max_plan_retries + 1):
            if self.trace is not None:
                self.trace.write("plan_attempt_start", attempt=attempt, error_context=_slim_error_context(error_context))

            exec_log: List[Dict[str, Any]] = []
            plan: Optional[Dict[str, Any]] = None
            results: Optional[Dict[str, Any]] = None
            try:
                self._progress(f"planning (attempt {attempt + 1}/{self.settings.max_plan_retries + 1})...")
                plan = self._generate_plan(user_prompt, error_context=error_context)

                self._progress("checking OpenCyc KB...")
                results = self._execute_plan(plan, exec_log=exec_log)

                answer_signal = self._extract_answer_signal(plan, results)
                if answer_signal is None:
                    raise NoAnswerError("OpenCyc returned no bindings for the final query.")

                if self.trace is not None:
                    self.trace.write("plan_attempt_succeeded", attempt=attempt)
                return plan, results, exec_log, answer_signal

            except (PlanValidationError, PlanExecutionError, NoAnswerError, CycBridgeError, OllamaError, Exception) as e:
                last_exception = e

                error_context = {
                    "attempt": attempt,
                    "error_type": e.__class__.__name__,
                    "error": str(e),
                    "plan": plan,
                    "execution_log": exec_log,
                    "cyc_results": results,
                }

                if self.trace is not None:
                    self.trace.write(
                        "plan_attempt_failed",
                        attempt=attempt,
                        error_type=e.__class__.__name__,
                        error=str(e),
                    )

        raise RuntimeError(
            f"Failed after {self.settings.max_plan_retries + 1} attempts. Last error: {last_exception}"
        )

    def _generate_plan(self, user_prompt: str, error_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        allow_converse = bool(self.allow_converse)
        prompt = user_prompt if error_context is None else _augment_prompt_with_error(user_prompt, error_context)

        messages = build_planner_messages(
            ctx=self.ctx,
            user_prompt=prompt,
            conversation=self.conversation,
            allow_converse=allow_converse,
        )

        plan = self._llm_chat_json(kind="planner", messages=messages, schema=plan_json_schema())

        if not isinstance(plan, dict):
            raise PlanValidationError(f"Planner returned non-object JSON: {type(plan)}")

        # Force session_mt to our current mt
        plan["session_mt"] = self.ctx.session_mt

        actions = plan.get("actions", [])
        if not isinstance(actions, list) or not actions:
            raise PlanValidationError("Planner returned empty/non-list 'actions'")

        normalized_actions: List[Dict[str, Any]] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            t = (a.get("type") or "").strip()
            if not t:
                continue

            if t == "converse" and not allow_converse:
                continue

            # Normalize mt to session mt for all non-converse actions
            if t != "converse":
                a["mt"] = self.ctx.session_mt

            # Validate/normalize action-specific fields
            normalized_actions.append(self._validate_and_normalize_action(a))

        if not normalized_actions:
            raise PlanValidationError("All actions were filtered out during normalization.")

        # Must end with ask_var/ask_true so the final value comes from OpenCyc
        final_type = (normalized_actions[-1].get("type") or "").strip()
        if final_type not in ("ask_var", "ask_true"):
            raise PlanValidationError("Plan must end with an ask_var or ask_true action.")

        # Ensure ensure_term constants are not placeholders and are actually used
        self._validate_ensure_terms_used(normalized_actions)

        plan["actions"] = normalized_actions
        return plan

    def _validate_and_normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        t = (action.get("type") or "").strip()

        if t == "ensure_term":
            name = (action.get("name") or "").strip()
            if not name:
                raise PlanValidationError("ensure_term missing 'name'")
            bare = name[2:] if name.startswith("#$") else name
            if bare.lower().startswith("ensure_"):
                raise PlanValidationError(
                    f"ensure_term name '{name}' looks like a placeholder. Use the actual constant name (CamelCase), not 'ensure_*'."
                )
            action["name"] = name
            return action

        if t == "assert":
            sentence = (action.get("sentence") or "").strip()
            if not sentence:
                raise PlanValidationError("assert missing 'sentence'")
            if not _is_single_toplevel_cycl_form(sentence):
                raise PlanValidationError(
                    f"assert sentence must be fully parenthesized CycL, got: {sentence[:120]}"
                )
            action["sentence"] = sentence
            return action

        if t == "ask_true":
            query = (action.get("query") or "").strip()
            if not query:
                raise PlanValidationError("ask_true missing 'query'")
            if not _is_single_toplevel_cycl_form(query):
                raise PlanValidationError(
                    f"ask_true query must be fully parenthesized CycL, got: {query[:120]}"
                )
            action["query"] = query
            return action

        if t == "ask_var":
            query = (action.get("query") or "").strip()
            var = (action.get("var") or "?X").strip() or "?X"
            limit = action.get("limit")
            if limit is None:
                limit = self.settings.default_bindings_limit
            else:
                try:
                    limit = int(limit)
                except Exception:
                    limit = self.settings.default_bindings_limit
            if limit <= 0:
                limit = self.settings.default_bindings_limit

            if not query:
                raise PlanValidationError("ask_var missing 'query'")
            if not _is_single_toplevel_cycl_form(query):
                raise PlanValidationError(
                    f"ask_var query must be fully parenthesized CycL, got: {query[:120]}"
                )
            if not var.startswith("?"):
                raise PlanValidationError(f"ask_var var must start with '?', got: {var}")
            if var not in query:
                raise PlanValidationError(
                    f"ask_var var '{var}' must appear in query. query={query[:160]}"
                )

            action["query"] = query
            action["var"] = var
            action["limit"] = limit
            return action

        if t == "converse":
            subl = (action.get("subl") or "").strip()
            if not subl:
                raise PlanValidationError("converse missing 'subl'")
            action["subl"] = subl
            return action

        raise PlanValidationError(f"Unknown action type: {t}")

    def _validate_ensure_terms_used(self, actions: List[Dict[str, Any]]) -> None:
        # For each ensure_term, require that its constant is referenced in a later assert/query.
        for i, a in enumerate(actions):
            if (a.get("type") or "").strip() != "ensure_term":
                continue
            name = (a.get("name") or "").strip()
            if not name:
                continue
            norm = _normalize_constant_ref(name)
            used = False
            for b in actions[i + 1 :]:
                t = (b.get("type") or "").strip()
                hay = ""
                if t == "assert":
                    hay = str(b.get("sentence") or "")
                elif t in ("ask_true", "ask_var"):
                    hay = str(b.get("query") or "")
                if norm and norm in hay:
                    used = True
                    break
            if not used:
                raise PlanValidationError(
                    f"ensure_term constant {norm} is never used later in the plan. Avoid unused placeholders."
                )

    def _execute_plan(self, plan: Dict[str, Any], exec_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "session_mt": self.ctx.session_mt,
            "actions": [],
        }

        actions = plan.get("actions", []) or []
        for idx, action in enumerate(actions):
            atype = (action.get("type") or "").strip()
            mt = self.ctx.session_mt
            entry: Dict[str, Any] = {"i": idx, "type": atype, "mt": mt}

            if self.trace is not None:
                self.trace.write("cyc_action_start", i=idx, type=atype, action=action)

            try:
                if atype == "ensure_term":
                    name = (action.get("name") or "").strip()
                    if not name:
                        entry["error"] = "Missing name"
                        results["actions"].append(entry)
                        exec_log.append(entry)
                        raise PlanExecutionError("ensure_term missing name")

                    self._progress(f"ensuring term: {name}")
                    exists = self.cyc.constant_exists(name)
                    entry["name"] = name
                    entry["exists"] = exists
                    if not exists:
                        created = self.cyc.create_constant(name)
                        entry["created"] = created
                    results["actions"].append(entry)
                    exec_log.append(entry)
                    continue

                if atype == "assert":
                    sentence = (action.get("sentence") or "").strip()
                    if not sentence:
                        entry["error"] = "Missing sentence"
                        results["actions"].append(entry)
                        exec_log.append(entry)
                        raise PlanExecutionError("assert missing sentence")

                    self._progress("asserting fact in session MT...")
                    self.cyc.assert_sentence(mt=mt, sentence=sentence)
                    entry["sentence"] = sentence
                    entry["ok"] = True
                    results["actions"].append(entry)
                    exec_log.append(entry)
                    continue

                if atype == "ask_true":
                    query = (action.get("query") or "").strip()
                    if not query:
                        entry["error"] = "Missing query"
                        results["actions"].append(entry)
                        exec_log.append(entry)
                        raise PlanExecutionError("ask_true missing query")

                    self._progress("querying OpenCyc...")
                    ans = self.cyc.ask_true(mt=mt, query=query)
                    entry["query"] = query
                    entry["answer"] = ans
                    results["actions"].append(entry)
                    exec_log.append(entry)
                    continue

                if atype == "ask_var":
                    query = (action.get("query") or "").strip()
                    var = (action.get("var") or "?X").strip() or "?X"
                    limit = action.get("limit")
                    if limit is None:
                        limit = self.settings.default_bindings_limit
                    else:
                        try:
                            limit = int(limit)
                        except Exception:
                            limit = self.settings.default_bindings_limit

                    if not query:
                        entry["error"] = "Missing query"
                        results["actions"].append(entry)
                        exec_log.append(entry)
                        raise PlanExecutionError("ask_var missing query")

                    self._progress("querying OpenCyc...")
                    bindings = self.cyc.ask_var(mt=mt, query=query, var=var, limit=limit)
                    entry["query"] = query
                    entry["var"] = var
                    entry["limit"] = limit
                    entry["bindings"] = bindings
                    results["actions"].append(entry)
                    exec_log.append(entry)
                    continue

                if atype == "converse":
                    subl = (action.get("subl") or "").strip()
                    if not subl:
                        entry["error"] = "Missing subl"
                        results["actions"].append(entry)
                        exec_log.append(entry)
                        raise PlanExecutionError("converse missing subl")
                    self._progress("running SubL...")
                    out = self.cyc.converse(subl=subl)
                    entry["subl"] = subl
                    entry["result"] = out
                    results["actions"].append(entry)
                    exec_log.append(entry)
                    continue

                entry["error"] = f"Unknown action type: {atype}"
                results["actions"].append(entry)
                exec_log.append(entry)
                raise PlanExecutionError(entry["error"])

            finally:
                if self.trace is not None:
                    self.trace.write("cyc_action_end", i=idx, type=atype, entry=entry)

        return results

    def _extract_answer_signal(self, plan: Dict[str, Any], cyc_results: Dict[str, Any]) -> Optional[AnswerSignal]:
        actions = plan.get("actions", []) or []
        if not actions:
            return None
        final_idx = len(actions) - 1
        final = actions[-1]
        ftype = (final.get("type") or "").strip()
        if ftype not in ("ask_var", "ask_true"):
            return None

        res_actions = cyc_results.get("actions", []) or []
        res_entry: Optional[Dict[str, Any]] = None
        for e in res_actions:
            if e.get("i") == final_idx and (e.get("type") or "").strip() == ftype:
                res_entry = e
                break
        if res_entry is None:
            return None

        if ftype == "ask_true":
            if "answer" not in res_entry:
                return None
            ans = bool(res_entry.get("answer"))
            return AnswerSignal(
                action_index=final_idx,
                action_type=ftype,
                answer_text="true" if ans else "false",
                raw_value=ans,
                query=str(res_entry.get("query") or ""),
            )

        bindings = res_entry.get("bindings", [])
        if not isinstance(bindings, list) or not bindings:
            return None
        answer_text = "\n".join(str(x) for x in bindings if x is not None and str(x).strip() != "")
        if not answer_text.strip():
            return None
        return AnswerSignal(
            action_index=final_idx,
            action_type=ftype,
            answer_text=answer_text,
            raw_value=bindings,
            query=str(res_entry.get("query") or ""),
        )

    # -----------------------------
    # Answer generation
    # -----------------------------

    def _generate_answer(
        self,
        user_prompt: str,
        cyc_results: Dict[str, Any],
        exec_log: List[Dict[str, Any]],
        answer_signal: AnswerSignal,
    ) -> Dict[str, Any]:
        authoritative_answer = answer_signal.answer_text

        payload = {
            "cyc_results": cyc_results,
            "execution_log": exec_log,
            "final_action": {
                "action_index": answer_signal.action_index,
                "action_type": answer_signal.action_type,
                "query": answer_signal.query,
                "raw_value": answer_signal.raw_value,
            },
        }

        messages = build_answer_messages(
            user_prompt=user_prompt,
            cyc_results=payload,
            authoritative_answer=authoritative_answer,
            conversation=self.conversation,
        )

        ans = self._llm_chat_json(kind="answerer", messages=messages, schema=answer_json_schema())

        if not isinstance(ans, dict) or "answer" not in ans:
            raise OllamaError(f"Answerer returned invalid JSON: {ans}")

        # Enforce list types
        ans["cyc_evidence"] = list(ans.get("cyc_evidence", []))
        ans["limitations"] = list(ans.get("limitations", []))

        # Hard guard: never allow the answerer to substitute a different answer.
        if str(ans.get("answer", "")).strip() != authoritative_answer.strip():
            ans["limitations"].append(
                "Answer text was corrected to match the authoritative OpenCyc-derived value."
            )
            ans["answer"] = authoritative_answer

        return ans


def _slim_error_context(error_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not error_context:
        return None
    slim = {
        "attempt": error_context.get("attempt"),
        "error_type": error_context.get("error_type"),
        "error": error_context.get("error"),
    }
    # Add last action if present
    exec_log = error_context.get("execution_log") or []
    if exec_log:
        slim["last_action"] = exec_log[-1]
    # Add final query result if present
    cyc_results = error_context.get("cyc_results") or {}
    if isinstance(cyc_results, dict) and cyc_results.get("actions"):
        slim["last_cyc_action"] = cyc_results.get("actions")[-1]
    return slim


def _augment_prompt_with_error(user_prompt: str, error_context: Dict[str, Any]) -> str:
    # Keep error payload compact; planner only needs the gist.
    slim = _slim_error_context(error_context) or {}
    return f"""{user_prompt}

PREVIOUS PLAN FAILED.
Error context (JSON):
{json.dumps(slim, indent=2)}

Please produce a corrected plan that:
- Uses fully parenthesized CycL in assert/query.
- Avoids creating placeholder constants (no ensure_* names).
- Ends with an ask_var/ask_true whose result directly contains the answer.
- If OpenCyc returned no bindings previously, assert the *minimal* missing facts in the session MT and re-query."""
