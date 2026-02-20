from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from .config import Settings
from .cyc_bridge_client import CycBridgeClient, CycBridgeError, SessionInfo

from . import corenlp_to_cycl as _E2C


class TranslationError(RuntimeError):
    pass


class NoAnswerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunResult:
    answer: str
    cyc_evidence: List[str]
    limitations: List[str]
    debug: Dict[str, Any]


_CYC_CONST_RE = re.compile(r"#\$[A-Za-z0-9][A-Za-z0-9_\-]*")
_INT_LIT_RE = re.compile(r"^-?\d+$")

# CycL quantifiers supported by your CycLParser grammar and used by corenlp_to_cycl.py.
_QUANTIFIER_HEADS: Set[str] = {
    "#$forAll",
    "#$thereExists",
    "#$thereExistExactly",
    "#$thereExistAtMost",
    "#$thereExistAtLeast",
}

# Variables produced by corenlp_to_cycl.py for WH-words.
_WH_VAR_PRIORITY: Sequence[str] = ("?Who", "?What", "?Which", "?Where", "?When")


_NL_STRING_PREDICATES: Sequence[str] = (
    "#$preferredNameString",
    "#$nameString",
    "#$termStrings",
    "#$termStrings-GuessedFromName",
    "#$acronymString",
    "#$initialismString",
    "#$abbreviationString-PN",
)


CycLTerm = _E2C.CycLTerm


def _extract_constants(text: str) -> List[str]:
    return _CYC_CONST_RE.findall(text or "")


def _is_int_literal(tok: str) -> bool:
    return bool(_INT_LIT_RE.match((tok or "").strip()))


def _split_top_level_sexp(expr: str) -> List[str]:
    """Split a single top-level CycL S-expression into its top-level elements.

    Example: '(#$P #$A 79)' -> ['#$P', '#$A', '79']
    Nested subexpressions are kept as single tokens.
    """
    expr = (expr or "").strip()
    if not expr.startswith("(") or not expr.endswith(")"):
        return []
    inner = expr[1:-1]

    tokens: List[str] = []
    buf: List[str] = []
    depth = 0
    in_str = False
    esc = False

    def flush() -> None:
        s = "".join(buf).strip()
        if s:
            tokens.append(s)
        buf.clear()

    for ch in inner:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            buf.append(ch)
            continue

        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue

        if ch.isspace() and depth == 0:
            flush()
            continue

        buf.append(ch)

    flush()
    return tokens


def _find_head_arities(expr: str, target_head: str) -> Set[int]:
    arities: Set[int] = set()
    _find_head_arities_rec(expr, target_head, arities)
    return arities


def _find_head_arities_rec(expr: str, target_head: str, out: Set[int]) -> None:
    toks = _split_top_level_sexp(expr)
    if not toks:
        return
    head = toks[0]
    if head == target_head:
        out.add(max(0, len(toks) - 1))
    for tok in toks[1:]:
        if tok.startswith("(") and tok.endswith(")"):
            _find_head_arities_rec(tok, target_head, out)


def _collect_vars(term: CycLTerm) -> Set[str]:
    if isinstance(term, str):
        return {term} if term.startswith("?") else set()
    out: Set[str] = set()
    for a in term:
        out |= _collect_vars(a)
    return out


def _free_vars(term: CycLTerm, bound: Optional[Set[str]] = None) -> Set[str]:
    bound2: Set[str] = set(bound or set())
    if isinstance(term, str):
        if term.startswith("?") and term not in bound2:
            return {term}
        return set()

    if not term:
        return set()

    head = term[0]
    if isinstance(head, str) and head in _QUANTIFIER_HEADS and len(term) >= 3:
        var = term[1]
        body = term[2]
        if isinstance(var, str) and var.startswith("?"):
            bound2 = set(bound2)
            bound2.add(var)
        return _free_vars(body, bound2)

    out: Set[str] = set()
    for a in term:
        out |= _free_vars(a, bound2)
    return out


def _drop_quantifier(term: CycLTerm, var: str) -> CycLTerm:
    """Remove any quantifier binding `var` (thereExists/forAll/etc), making `var` free."""
    if isinstance(term, str):
        return term
    if not term:
        return term

    head = term[0]
    if isinstance(head, str) and head in _QUANTIFIER_HEADS and len(term) >= 3:
        q_var = term[1]
        body = term[2]
        if q_var == var:
            return _drop_quantifier(body, var)
        return [head, q_var, _drop_quantifier(body, var)]

    return [_drop_quantifier(a, var) for a in term]


def _close_free_vars(term: CycLTerm, *, keep_free: Set[str]) -> CycLTerm:
    free = _free_vars(term)
    to_bind = sorted(v for v in free if v not in keep_free)
    out = term
    for v in to_bind:
        out = ["#$thereExists", v, out]
    return out


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    # Heuristic: leading auxiliaries often mark yes/no questions.
    first = re.split(r"\s+", t, maxsplit=1)[0].lower()
    return first in {
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "has",
        "have",
        "had",
        "may",
        "might",
        "must",
    }


def _unquote_cyc_string(s: str) -> str:
    t = (s or "").strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        inner = t[1:-1]
        # Unescape minimal Cyc string escapes.
        inner = inner.replace("\\\\", "\\").replace('\\"', '"')
        return inner
    return t


class Orchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cyc = CycBridgeClient(settings.cyc_bridge_base_url)

        self.session_info = self._init_session()
        self.debug: bool = False

        # Optional progress callback (UI layer)
        self._progress_cb: Optional[Callable[[str], None]] = None

        # CoreNLP client
        self._nlp = _E2C.CoreNLPServerClient(
            base_url=settings.corenlp_base_url,
            timeout_sec=int(settings.http_timeout_sec),
        )

        # Cyc-backed lexicon/scoring (used by corenlp_to_cycl translator)
        lex_bridge: Optional[_E2C.CycBridgeClient] = None
        if settings.use_cyc_lexicon or settings.use_cyc_scorer:
            lex_bridge = _E2C.CycBridgeClient(
                base_url=settings.cyc_bridge_base_url,
                timeout_sec=int(settings.http_timeout_sec),
            )

        lexicon = _E2C.CycLexicon(
            bridge=lex_bridge if settings.use_cyc_lexicon else None,
            lex_mt=settings.cyc_lexicon_mt,
            lex_limit=int(settings.cyc_lex_limit),
        )

        self._translator = _E2C.CycLTranslator(
            lexicon=lexicon,
            query_mt=settings.cyc_query_mt,
            enable_scorer=bool(settings.use_cyc_scorer),
        )

        self._nl_mt = settings.cyc_nl_mt.strip() or settings.cyc_lexicon_mt

    # -----------------------------
    # Lifecycle / UI hooks
    # -----------------------------

    def close(self) -> None:
        return

    def set_progress_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._progress_cb = cb

    def _progress(self, msg: str) -> None:
        if self._progress_cb is not None:
            self._progress_cb(msg)

    def set_debug(self, enabled: bool) -> None:
        self.debug = bool(enabled)

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

    # -----------------------------
    # Public API
    # -----------------------------

    def handle_user_prompt(self, user_prompt: str) -> RunResult:
        debug: Dict[str, Any] = {
            "session": {
                "session_id": self.session_info.session_id,
                "session_mt": self.session_info.session_mt,
                "genl_mt": self.session_info.genl_mt,
            },
            "prompt": user_prompt,
        }

        self._progress("parsing English with CoreNLP and composing CycL...")
        query_type, query_str, query_var = self._translate_to_query(user_prompt, debug_out=debug)

        self._progress("querying OpenCyc...")
        exec_log: List[Dict[str, Any]] = []
        raw_value = self._execute_query_with_repairs(
            query_type=query_type,
            query=query_str,
            var=query_var,
            limit=int(self.settings.default_bindings_limit),
            exec_log=exec_log,
        )

        self._progress("rendering result...")
        answer, evidence, limitations = self._format_result(
            query_type=query_type,
            query=query_str,
            var=query_var,
            raw_value=raw_value,
        )

        debug["query"] = {
            "query_type": query_type,
            "query": query_str,
            "var": query_var,
        }
        debug["execution_log"] = exec_log
        debug["raw_value"] = raw_value

        return RunResult(
            answer=answer,
            cyc_evidence=evidence,
            limitations=limitations,
            debug=debug,
        )

    # -----------------------------
    # Translation (English -> CycL)
    # -----------------------------

    def _translate_to_query(self, user_prompt: str, *, debug_out: Dict[str, Any]) -> Tuple[str, str, str]:
        try:
            ann = self._nlp.annotate(user_prompt)
        except Exception as e:
            raise TranslationError(f"CoreNLP annotate() failed: {e}") from e

        try:
            term: CycLTerm = self._translator.translate_annotation_term(ann)
        except Exception as e:
            raise TranslationError(f"CycL composition failed: {e}") from e

        debug_out["raw_cycl_term"] = term
        debug_out["raw_cycl"] = _E2C.cycl_to_string(term)

        is_question = _looks_like_question(user_prompt)

        # Determine if this is a WH-question with a target variable.
        vars_in_term = _collect_vars(term)
        wh_var = ""
        for v in _WH_VAR_PRIORITY:
            if v in vars_in_term:
                wh_var = v
                break

        if wh_var:
            # ask_var: make WH variable free and close any other free variables.
            query_type = "ask_var"
            query_var = wh_var
            t2 = _drop_quantifier(term, wh_var)
            t3 = _close_free_vars(t2, keep_free={wh_var})
            query_str = _E2C.cycl_to_string(t3)
            debug_out["cycl_rewrite"] = {
                "wh_var": wh_var,
                "after_drop_quantifier": _E2C.cycl_to_string(t2),
                "after_close_free_vars": query_str,
                "free_vars_after": sorted(_free_vars(t3)),
            }
            return query_type, query_str, query_var

        # Default: yes/no or declarative -> ask_true (closed).
        query_type = "ask_true" if is_question else "ask_true"
        t2 = _close_free_vars(term, keep_free=set())
        query_str = _E2C.cycl_to_string(t2)
        debug_out["cycl_rewrite"] = {
            "wh_var": None,
            "after_close_free_vars": query_str,
            "free_vars_after": sorted(_free_vars(t2)),
        }
        return query_type, query_str, ""

    # -----------------------------
    # Execution + repairs
    # -----------------------------

    def _execute_query_with_repairs(
        self,
        *,
        query_type: str,
        query: str,
        var: str,
        limit: int,
        exec_log: List[Dict[str, Any]],
    ) -> Any:
        repairs = 0

        while True:
            try:
                if query_type == "ask_true":
                    ans = self.cyc.ask_true(mt=self.session_info.session_mt, query=query)
                    exec_log.append({"type": "ask_true", "mt": self.session_info.session_mt, "query": query, "answer": ans})
                    return bool(ans)

                if query_type == "ask_var":
                    if not var or not var.startswith("?"):
                        raise TranslationError(f"ask_var requires a variable like '?X', got: {var!r}")
                    bindings = self.cyc.ask_var(
                        mt=self.session_info.session_mt,
                        query=query,
                        var=var,
                        limit=int(limit),
                    )
                    exec_log.append(
                        {
                            "type": "ask_var",
                            "mt": self.session_info.session_mt,
                            "query": query,
                            "var": var,
                            "limit": int(limit),
                            "bindings": bindings,
                        }
                    )
                    return bindings

                raise RuntimeError(f"Unknown query_type: {query_type}")

            except CycBridgeError as e:
                msg = e.server_message
                exec_log.append({"type": "cyc_error", "message": msg})

                # Try to auto-repair missing constants.
                missing = sorted(set(_extract_constants(msg)))
                if not missing:
                    raise

                safe_missing: List[str] = []
                for tok in missing:
                    bare = tok[2:] if tok.startswith("#$") else tok
                    if not bare:
                        continue
                    try:
                        exists = self.cyc.constant_exists(bare)
                    except CycBridgeError:
                        # If we can't check existence, treat it as non-repairable.
                        continue
                    if exists:
                        continue
                    if self._safe_to_autocreate_constant(tok):
                        safe_missing.append(tok)

                if not safe_missing:
                    raise

                repairs += 1
                if repairs > int(self.settings.max_missing_term_repairs):
                    raise RuntimeError(
                        f"Exceeded MAX_MISSING_TERM_REPAIRS={self.settings.max_missing_term_repairs}. Last error: {msg}"
                    ) from e

                self._progress("defining missing constants in session MT...")
                self._define_missing_constants(safe_missing, query=query, exec_log=exec_log)
                # retry

    def _safe_to_autocreate_constant(self, const_tok: str) -> bool:
        if not const_tok.startswith("#$") or len(const_tok) < 3:
            return False
        c0 = const_tok[2]
        return c0.isupper() or c0.isdigit()

    def _define_missing_constants(self, missing_constants: Sequence[str], *, query: str, exec_log: List[Dict[str, Any]]) -> None:
        seen: Set[str] = set()

        for tok in missing_constants:
            if tok in seen:
                continue
            seen.add(tok)

            bare = tok[2:] if tok.startswith("#$") else tok
            if not bare:
                continue

            # Ensure the constant exists.
            try:
                created_name = self.cyc.create_constant(bare)
                exec_log.append({"type": "ensure_term", "name": bare, "created": created_name})
            except CycBridgeError as e:
                exec_log.append({"type": "ensure_term", "name": bare, "error": e.server_message})
                continue

            # Assert a minimal type in the session MT (helps Cyc accept some queries).
            try:
                arities = _find_head_arities(query, tok)
                if arities:
                    arity = max(arities)
                    pred_type = {
                        1: "#$UnaryPredicate",
                        2: "#$BinaryPredicate",
                        3: "#$TernaryPredicate",
                        4: "#$QuaternaryPredicate",
                    }.get(arity, "#$Predicate")
                    sent = f"(#$isa {tok} {pred_type})"
                else:
                    sent = f"(#$isa {tok} #$Thing)"

                self.cyc.assert_sentence(mt=self.session_info.session_mt, sentence=sent)
                exec_log.append({"type": "assert", "mt": self.session_info.session_mt, "sentence": sent})
            except CycBridgeError as e:
                exec_log.append({"type": "assert", "mt": self.session_info.session_mt, "sentence": sent, "error": e.server_message})

    # -----------------------------
    # Natural-language rendering
    # -----------------------------

    def _term_to_english(self, term_str: str) -> str:
        t = (term_str or "").strip()
        if not t:
            return t

        if _is_int_literal(t):
            return t
        if t.startswith('"') and t.endswith('"'):
            return _unquote_cyc_string(t)

        if not self.settings.use_cyc_nl:
            return t

        # Constants are the easy/common case.
        if t.startswith("#$"):
            for pred in _NL_STRING_PREDICATES:
                q = f"({pred} {t} ?S)"
                try:
                    vals = self.cyc.ask_var(mt=self._nl_mt, query=q, var="?S", limit=1)
                except Exception:
                    vals = []
                if vals:
                    s = _unquote_cyc_string(vals[0])
                    if s:
                        return s

        return t

    def _format_result(
        self,
        *,
        query_type: str,
        query: str,
        var: str,
        raw_value: Any,
    ) -> Tuple[str, List[str], List[str]]:
        evidence: List[str] = []
        limitations: List[str] = []

        evidence.append(f"{query_type}: {query}")

        if query_type == "ask_true":
            ans = bool(raw_value)
            return ("true" if ans else "false"), evidence, limitations

        # ask_var
        bindings = raw_value if isinstance(raw_value, list) else []
        if not bindings:
            limitations.append("OpenCyc returned no bindings for the translated query.")
            return "(no bindings)", evidence, limitations

        rendered: List[str] = []
        for b in bindings:
            rendered.append(self._term_to_english(str(b)))

        # If the NL rendering differs from raw, we can optionally show both in debug mode.
        if self.debug:
            evidence.append(f"var={var}  count={len(bindings)}")

        return "\n".join(rendered), evidence, limitations
