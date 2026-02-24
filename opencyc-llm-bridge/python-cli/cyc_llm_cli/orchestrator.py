from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from .config import Settings
from .cyc_bridge_client import CycBridgeClient, CycBridgeError, SessionInfo

from . import corenlp_to_cycl as _E2C

# Debug: uncomment the following line to verify which module file is loaded.
# print(f"[DEBUG] corenlp_to_cycl loaded from: {_E2C.__file__}")


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

_QUANTIFIER_HEADS: Set[str] = {
    "#$forAll", "#$thereExists",
    "#$thereExistExactly", "#$thereExistAtMost", "#$thereExistAtLeast",
}

_WH_VAR_PRIORITY: Sequence[str] = ("?Who", "?What", "?Which", "?Where", "?When")

_WH_COPULAR_RE = re.compile(
    r"^\s*(who|what|which)\s+(is|are|was|were|am)\s+(.+?)[\s?!.]*$",
    re.IGNORECASE,
)

_NL_STRING_PREDICATES: Sequence[str] = (
    "#$preferredNameString", "#$nameString", "#$termStrings",
    "#$termStrings-GuessedFromName", "#$acronymString",
    "#$initialismString", "#$abbreviationString-PN",
)

_ABSTRACT_TYPE_BLACKLIST: Set[str] = {
    "Thing", "Individual", "TemporalThing", "TemporallyExistingThing",
    "SomethingExisting", "PartiallyIntangible", "PartiallyIntangibleIndividual",
    "Agent-Generic", "Agent-Underspecified", "Trajector-Underspecified",
    "Location-Underspecified", "CycLTerm", "CycLReifiableDenotationalTerm",
    "CycLConstant", "CycLExpression", "CycLReifiableNonAtomicTerm",
}

_ENTITY_INFO_PREDICATES: Sequence[Tuple[str, str]] = (
    ("(#$comment {C} ?V)", "description"),
    ("(#$occupation {C} ?V)", "occupation"),
    ("(#$residesInRegion {C} ?V)", "resides in"),
    ("(#$birthDate {C} ?V)", "born"),
    ("(#$dateOfDeath {C} ?V)", "died"),
    ("(#$spouse {C} ?V)", "spouse"),
    ("(#$affiliatedWith {C} ?V)", "affiliated with"),
    ("(#$citizens {C} ?V)", "citizen of"),
)

# ------------------------------------------------------------------
# LLM KB augmentation: prompt templates and schemas
# ------------------------------------------------------------------

# JSON schema for dual-path LLM output.
_LLM_AUG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Simple English declarative sentences (X is a Y) for pipeline translation.",
        },
        "assertions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Direct CycL assertion strings for predicates.",
        },
    },
    "required": ["sentences", "assertions"],
    "additionalProperties": False,
}

_LLM_AUG_ENTITY_SYSTEM = """\
You are a knowledge-base assistant. You supply FACTUAL information to fill gaps in
the OpenCyc knowledge base.

Return a JSON object with TWO lists:

1. "sentences" — Simple English declarative sentences about the entity.
   Use ONLY the pattern "NAME is a TYPE." or "NAME is an TYPE."
   ONE fact per sentence.  Use the entity's common English name.
   Examples:
     "Bill Clinton is a president."
     "Bill Clinton is a politician."
     "Bill Clinton is an American."
   These will be parsed by an NLP pipeline to resolve the right Cyc constants.

2. "assertions" — Direct CycL assertion strings for facts the NLP pipeline cannot handle.
   All constants must be #$-prefixed CamelCase.
   String arguments use double-quotes.
   USE THESE PREDICATES (they exist in OpenCyc):
     (#$comment <entity> "<text description>")
     (#$isa <entity> <Collection>)
     (#$genls <SubCollection> <SuperCollection>)
     (#$occupation <person> <OccupationType>)
     (#$birthDate <entity> <year-integer>)
     (#$dateOfDeath <entity> <year-integer>)
     (#$spouse <person1> <person2>)
     (#$residesInRegion <agent> <region>)
     (#$affiliatedWith <agent> <organization>)
     (#$citizens <person> <country>)
   When introducing a NEW constant, also add a (#$genls NewCollection BroaderCollection).
   ALWAYS include at least one (#$comment ...) with a short natural-language description.
   DO NOT wrap in #$ist.

Return ONLY the JSON object, nothing else:
{"sentences": ["...","..."], "assertions": ["(...)","(...)"]}

Only supply facts you are confident are true. 3-8 items per list."""

_LLM_AUG_GENERAL_SYSTEM = """\
You are a knowledge-base assistant. A query to the OpenCyc knowledge base
returned no results. You supply factual information to fill the gap.

Return a JSON object with TWO lists:

1. "sentences" — Simple English declarative "X is a Y." sentences.
   These are parsed by NLP to resolve Cyc constants automatically.

2. "assertions" — Direct CycL assertion strings for specific predicates.
   All constants must be #$-prefixed CamelCase.
   PREDICATES: #$isa, #$genls, #$comment, #$occupation, #$birthDate,
   #$spouse, #$residesInRegion, #$affiliatedWith, #$citizens.
   When introducing a new constant, add (#$genls ...) too.
   DO NOT wrap in #$ist.

Return ONLY: {"sentences": ["...","..."], "assertions": ["(...)","(...)"]}

Focus on the specific facts needed to answer the query. 3-8 items per list."""


CycLTerm = _E2C.CycLTerm


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _extract_constants(text: str) -> List[str]:
    return _CYC_CONST_RE.findall(text or "")


def _is_int_literal(tok: str) -> bool:
    return bool(_INT_LIT_RE.match((tok or "").strip()))


def _split_top_level_sexp(expr: str) -> List[str]:
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
    if toks[0] == target_head:
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
        return {term} if term.startswith("?") and term not in bound2 else set()
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
    first = re.split(r"\s+", t, maxsplit=1)[0].lower()
    return first in {
        "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "will", "would", "should",
        "has", "have", "had", "may", "might", "must",
    }


def _unquote_cyc_string(s: str) -> str:
    t = (s or "").strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        inner = t[1:-1].replace("\\\\", "\\").replace('\\"', '"')
        return inner
    return t


def _is_abstract_type(binding_str: str) -> bool:
    b = (binding_str or "").strip()
    if not b:
        return True
    if b.startswith("("):
        return True
    bare = b[2:] if b.startswith("#$") else b
    if bare in _ABSTRACT_TYPE_BLACKLIST:
        return True
    if bare.endswith("-Underspecified"):
        return True
    return False


# Tokens that should NOT get a #$ prefix during CycL normalization.
_NON_CONST_WORDS: Set[str] = {
    "and", "or", "not", "if", "iff", "the", "a", "an", "of", "in", "is",
    "are", "was", "were", "has", "have", "be", "to", "for", "with", "on",
    "at", "from", "by", "as", "it", "its", "that", "this", "true", "false",
    "nil", "NIL", "T",
}


def _normalize_cycl_assertion(raw: str) -> str:
    """Add #$ prefixes to bare Cyc constant names in a CycL assertion string."""
    s = raw.strip()
    if not s:
        return s
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == '"':
            j = i + 1
            while j < n:
                if s[j] == '\\' and j + 1 < n:
                    j += 2
                    continue
                if s[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(s[i:j])
            i = j
            continue
        if ch == '#' and i + 1 < n and s[i + 1] == '$':
            j = i + 2
            while j < n and (s[j].isalnum() or s[j] in ('_', '-')):
                j += 1
            out.append(s[i:j])
            i = j
            continue
        if ch == '?':
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] in ('_', '-')):
                j += 1
            out.append(s[i:j])
            i = j
            continue
        if ch.isalpha():
            j = i
            while j < n and (s[j].isalnum() or s[j] in ('_', '-')):
                j += 1
            token = s[i:j]
            if token.lower() in _NON_CONST_WORDS:
                out.append(token)
            elif token.isdigit():
                out.append(token)
            else:
                out.append("#$" + token)
            i = j
            continue
        if ch.isdigit() or (ch == '-' and i + 1 < n and s[i + 1].isdigit()):
            j = i + 1 if ch == '-' else i
            while j < n and s[j].isdigit():
                j += 1
            out.append(s[i:j])
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

class Orchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cyc = CycBridgeClient(settings.cyc_bridge_base_url)

        self.session_info = self._init_session()
        self.debug: bool = False
        self._progress_cb: Optional[Callable[[str], None]] = None

        # CoreNLP client
        self._nlp = _E2C.CoreNLPServerClient(
            base_url=settings.corenlp_base_url,
            timeout_sec=int(settings.http_timeout_sec),
        )

        # Cyc-backed lexicon/scoring
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

        # Ollama client for LLM KB augmentation
        self._ollama: Optional[_E2C.OllamaClient] = None
        if settings.use_llm_kb_augmentation:
            client = _E2C.OllamaClient(
                base_url=settings.ollama_base_url,
                timeout_sec=int(settings.ollama_timeout_sec),
            )
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"{settings.ollama_base_url.rstrip('/')}/api/tags", method="GET",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp.read()
                self._ollama = client
            except Exception:
                self._ollama = None

    # ---- Lifecycle / UI hooks ----

    def close(self) -> None:
        return

    def set_progress_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._progress_cb = cb

    def _progress(self, msg: str) -> None:
        if self._progress_cb is not None:
            self._progress_cb(msg)

    def set_debug(self, enabled: bool) -> None:
        self.debug = bool(enabled)

    def _init_session(self) -> SessionInfo:
        session_id = uuid.uuid4().hex[:12]
        return self.cyc.ensure_session(
            session_id=session_id,
            comment=self.settings.session_mt_comment,
            genl_mt=self.settings.session_mt_genl,
        )

    # =============================
    # Public API
    # =============================

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

        wh_entity = debug.get("_wh_copular_entity")
        if wh_entity:
            return self._handle_entity_description(
                entity_const=wh_entity, user_prompt=user_prompt, debug=debug,
            )

        # Normal query execution.
        self._progress("querying OpenCyc...")
        exec_log: List[Dict[str, Any]] = []
        raw_value = self._execute_query_with_repairs(
            query_type=query_type, query=query_str, var=query_var,
            limit=int(self.settings.default_bindings_limit), exec_log=exec_log,
        )

        empty_result = (query_type == "ask_var" and (not raw_value or raw_value == []))
        if empty_result and self._ollama is not None:
            self._progress("no results — asking LLM to fill knowledge gap...")
            augmented = self._augment_kb_general(
                user_prompt=user_prompt, failed_query=query_str,
                query_var=query_var, exec_log=exec_log,
            )
            if augmented:
                self._progress("re-querying OpenCyc with augmented KB...")
                raw_value = self._execute_query_with_repairs(
                    query_type=query_type, query=query_str, var=query_var,
                    limit=int(self.settings.default_bindings_limit), exec_log=exec_log,
                )

        self._progress("rendering result...")
        answer, evidence, limitations = self._format_result(
            query_type=query_type, query=query_str, var=query_var, raw_value=raw_value,
        )

        debug["query"] = {"query_type": query_type, "query": query_str, "var": query_var}
        debug["execution_log"] = exec_log
        debug["raw_value"] = raw_value

        return RunResult(answer=answer, cyc_evidence=evidence, limitations=limitations, debug=debug)

    # =============================
    # Translation (English -> CycL)
    # =============================

    def _translate_to_query(self, user_prompt: str, *, debug_out: Dict[str, Any]) -> Tuple[str, str, str]:
        try:
            ann = self._nlp.annotate(user_prompt)
        except Exception as e:
            raise TranslationError(f"CoreNLP annotate() failed: {e}") from e

        wh_match = _WH_COPULAR_RE.match(user_prompt)
        if wh_match:
            name_part = wh_match.group(3).strip()
            if name_part:
                self._resolve_wh_entity(name_part, ann, debug_out)

        try:
            term: CycLTerm = self._translator.translate_annotation_term(ann)
        except Exception as e:
            raise TranslationError(f"CycL composition failed: {e}") from e

        debug_out["raw_cycl_term"] = term
        debug_out["raw_cycl"] = _E2C.cycl_to_string(term)
        debug_out["corenlp_to_cycl_file"] = getattr(_E2C, "__file__", "(unknown)")

        vars_in_term = _collect_vars(term)
        wh_var = ""
        for v in _WH_VAR_PRIORITY:
            if v in vars_in_term:
                wh_var = v
                break

        if wh_var:
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

        if wh_match and "_wh_copular_entity" in debug_out:
            entity = debug_out["_wh_copular_entity"]
            query_str = f"(#$and (#$isa ?What #$Collection) (#$isa {entity} ?What))"
            debug_out["cycl_rewrite"] = {
                "wh_var": "?What", "fallback": "wh_copular_surface_regex",
                "original_term": _E2C.cycl_to_string(term),
                "after_close_free_vars": query_str, "free_vars_after": ["?What"],
            }
            return "ask_var", query_str, "?What"

        t2 = _close_free_vars(term, keep_free=set())
        query_str = _E2C.cycl_to_string(t2)
        debug_out["cycl_rewrite"] = {
            "wh_var": None, "after_close_free_vars": query_str,
            "free_vars_after": sorted(_free_vars(t2)),
        }
        return "ask_true", query_str, ""

    def _resolve_wh_entity(self, name_text: str, ann: Any, debug_out: Dict[str, Any]) -> None:
        name_norm = re.sub(r"(?<=[A-Za-z])[-_](?=[A-Za-z])", " ", name_text)
        name_norm = re.sub(r"\s+", " ", name_norm).strip()
        if not name_norm:
            return
        parts = name_norm.split()
        caps = sum(1 for p in parts if p[:1].isupper())
        has_ner = False
        try:
            for sent in (ann or {}).get("sentences", []):
                for tok in sent.get("tokens", []):
                    w = (tok.get("word") or "").strip()
                    ner = (tok.get("ner") or "O").strip()
                    if w and ner not in ("O", "") and w.lower() in name_text.lower():
                        has_ner = True
                        break
        except Exception:
            pass
        if caps < 1 and not has_ner:
            return
        ner_tag = ""
        try:
            for sent in (ann or {}).get("sentences", []):
                for tok in sent.get("tokens", []):
                    ner = (tok.get("ner") or "O").strip()
                    if ner not in ("O", ""):
                        ner_tag = ner
                        break
        except Exception:
            pass
        subj_const = self._translator._choose_proper_constant(text=name_norm, ner=ner_tag)
        debug_out["_wh_copular_entity"] = subj_const

    # ============================================
    # Entity description (multi-strategy)
    # ============================================

    def _handle_entity_description(
        self, *, entity_const: str, user_prompt: str, debug: Dict[str, Any],
    ) -> RunResult:
        self._progress("describing entity via OpenCyc (multi-strategy)...")
        exec_log: List[Dict[str, Any]] = []
        entity_name = self._term_to_english(entity_const)

        sections, evidence, limitations = self._run_entity_strategies(entity_const, exec_log=exec_log)

        if not sections and self._ollama is not None:
            self._progress("no KB data — asking LLM for facts (pipeline + direct)...")
            augmented = self._augment_kb_entity(
                entity_const=entity_const, entity_name=entity_name, exec_log=exec_log,
            )
            if augmented:
                self._progress("re-running entity strategies with augmented KB...")
                sections, evidence, limitations = self._run_entity_strategies(
                    entity_const, exec_log=exec_log,
                )

        if not sections:
            limitations.append(
                f"OpenCyc has {entity_const} but returned no human-meaningful descriptions or types."
            )
            answer = f"{entity_name} (no detailed description available in OpenCyc)"
        else:
            header = entity_name if entity_name != entity_const else entity_const
            answer = f"{header}\n" + "\n".join(sections)

        debug["query"] = {
            "query_type": "entity_description", "entity_const": entity_const,
            "strategies": ["predicates", "filtered-isa", "llm-augmentation"],
        }
        debug["execution_log"] = exec_log

        return RunResult(answer=answer, cyc_evidence=evidence, limitations=limitations, debug=debug)

    def _run_entity_strategies(
        self, entity_const: str, *, exec_log: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        sections: List[str] = []
        evidence: List[str] = []
        limitations: List[str] = []

        facts = self._get_entity_facts(entity_const, exec_log=exec_log)
        for label, values in facts:
            rendered_vals = [self._term_to_english(v) for v in values[:5]]
            rendered_vals = [v for v in rendered_vals if v]
            if rendered_vals:
                if label == "description":
                    sections.insert(0, rendered_vals[0])
                else:
                    sections.append(f"{label.capitalize()}: {', '.join(rendered_vals)}")
                evidence.append(f"{label}: {len(values)} value(s)")

        filtered_isa = self._get_filtered_isa(entity_const, exec_log=exec_log)
        if filtered_isa:
            rendered_types = self._render_type_list(filtered_isa)
            if rendered_types:
                sections.append("Types: " + ", ".join(rendered_types))
                evidence.append(f"isa (filtered): {len(filtered_isa)} type(s)")

        return sections, evidence, limitations

    def _get_entity_facts(self, const: str, *, exec_log: List[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
        results: List[Tuple[str, List[str]]] = []
        for template, label in _ENTITY_INFO_PREDICATES:
            q = template.replace("{C}", const)
            try:
                vals = self.cyc.ask_var(mt=self.session_info.session_mt, query=q, var="?V", limit=10)
                exec_log.append({"type": "ask_var", "query": q, "bindings": vals})
            except Exception as e:
                exec_log.append({"type": "ask_var", "query": q, "error": str(e)})
                vals = []
            if vals:
                results.append((label, vals))
        return results

    def _get_filtered_isa(self, const: str, *, exec_log: List[Dict[str, Any]]) -> List[str]:
        q = f"(#$and (#$isa {const} ?What) (#$isa ?What #$Collection))"
        try:
            vals = self.cyc.ask_var(mt=self.session_info.session_mt, query=q, var="?What", limit=50)
            exec_log.append({"type": "ask_var", "query": q, "bindings": vals})
        except Exception as e:
            exec_log.append({"type": "ask_var", "query": q, "error": str(e)})
            vals = []
        return [v for v in vals if not _is_abstract_type(v)]

    def _render_type_list(self, types: List[str]) -> List[str]:
        rendered: List[str] = []
        seen: Set[str] = set()
        for t in types:
            if _is_abstract_type(t):
                continue
            eng = self._term_to_english(t)
            if not eng or eng in seen:
                continue
            seen.add(eng)
            rendered.append(eng)
        return rendered

    # ============================================
    # LLM KB Augmentation (dual-path)
    # ============================================

    def _call_ollama_dual(
        self, *, system: str, user: str, exec_log: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        """Call Ollama and return (english_sentences, cycl_assertions)."""
        if self._ollama is None:
            return [], []

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        options = {"temperature": 0.0, "num_predict": 2048}

        try:
            resp = self._ollama.chat_json(
                model=self.settings.ollama_model, messages=messages,
                schema=_LLM_AUG_SCHEMA, options=options,
            )
        except Exception:
            try:
                resp = self._ollama.chat_json(
                    model=self.settings.ollama_model, messages=messages,
                    schema=None, options=options,
                )
            except Exception as e2:
                exec_log.append({"type": "llm_call", "error": str(e2)})
                return [], []

        data = _E2C._extract_json_from_ollama_response(resp)

        # Extract English sentences.
        raw_sents = data.get("sentences") or []
        if not isinstance(raw_sents, list):
            raw_sents = []
        sentences = [s.strip() for s in raw_sents if isinstance(s, str) and s.strip()]

        # Extract and normalize direct CycL assertions.
        raw_asserts = data.get("assertions") or []
        if not isinstance(raw_asserts, list):
            raw_asserts = []
        direct: List[str] = []
        for item in raw_asserts:
            if not isinstance(item, str):
                continue
            s = _normalize_cycl_assertion(item.strip())
            if s.startswith("(") and s.endswith(")") and "#$" in s:
                direct.append(s)

        exec_log.append({
            "type": "llm_dual",
            "raw_sentences": [str(s) for s in raw_sents],
            "raw_assertions": [str(s) for s in raw_asserts],
            "valid_sentences": len(sentences),
            "valid_assertions": len(direct),
        })
        return sentences, direct

    def _translate_sentences_to_assertions(
        self, sentences: List[str], *, entity_hint: Optional[str], exec_log: List[Dict[str, Any]],
    ) -> List[str]:
        """Run English sentences through CoreNLP → translator → ground CycL assertions."""
        all_assertions: List[str] = []
        for sent_text in sentences:
            try:
                ann = self._nlp.annotate(sent_text)
            except Exception as e:
                exec_log.append({"type": "pipeline_translate", "sentence": sent_text, "error": str(e)})
                continue

            try:
                assertions = self._translator.translate_to_assertions(ann, entity_hint=entity_hint)
            except Exception as e:
                exec_log.append({"type": "pipeline_translate", "sentence": sent_text, "error": str(e)})
                continue

            exec_log.append({
                "type": "pipeline_translate",
                "sentence": sent_text,
                "assertions": assertions,
            })
            all_assertions.extend(assertions)

        return all_assertions

    def _assert_facts(
        self, assertions: List[str], *, source: str, exec_log: List[Dict[str, Any]],
    ) -> int:
        """Assert CycL sentences into the session MT, creating missing constants. Returns count."""
        mt = self.session_info.session_mt
        asserted = 0
        for sentence in assertions:
            consts = _extract_constants(sentence)
            for c in consts:
                bare = c[2:] if c.startswith("#$") else c
                if not bare:
                    continue
                try:
                    exists = self.cyc.constant_exists(bare)
                except CycBridgeError:
                    continue
                if exists:
                    continue
                if not self._safe_to_autocreate_constant(c):
                    continue
                try:
                    self.cyc.create_constant(bare)
                    exec_log.append({"type": "ensure_term", "name": bare, "source": source})
                except CycBridgeError as e:
                    exec_log.append({"type": "ensure_term", "name": bare, "error": e.server_message})
            try:
                self.cyc.assert_sentence(mt=mt, sentence=sentence)
                exec_log.append({"type": "assert", "mt": mt, "sentence": sentence, "source": source})
                asserted += 1
            except CycBridgeError as e:
                exec_log.append({"type": "assert", "mt": mt, "sentence": sentence, "error": e.server_message, "source": source})
        return asserted

    def _augment_kb_entity(
        self, *, entity_const: str, entity_name: str, exec_log: List[Dict[str, Any]],
    ) -> bool:
        user_msg = (
            f"The OpenCyc knowledge base has the entity {entity_const} "
            f'(name: "{entity_name}") but no detailed information.\n\n'
            f"Please supply factual information about this entity."
        )
        sentences, direct_assertions = self._call_ollama_dual(
            system=_LLM_AUG_ENTITY_SYSTEM, user=user_msg, exec_log=exec_log,
        )

        total = 0

        # Path 1: English sentences → CoreNLP → translate_to_assertions → assert
        if sentences:
            self._progress(f"translating {len(sentences)} LLM sentence(s) via NLP pipeline...")
            pipeline_assertions = self._translate_sentences_to_assertions(
                sentences, entity_hint=entity_const, exec_log=exec_log,
            )
            if pipeline_assertions:
                total += self._assert_facts(pipeline_assertions, source="llm-pipeline", exec_log=exec_log)

        # Path 2: Direct CycL assertions → normalize → assert
        if direct_assertions:
            self._progress(f"asserting {len(direct_assertions)} direct CycL assertion(s)...")
            total += self._assert_facts(direct_assertions, source="llm-direct", exec_log=exec_log)

        return total > 0

    def _augment_kb_general(
        self, *, user_prompt: str, failed_query: str, query_var: str,
        exec_log: List[Dict[str, Any]],
    ) -> bool:
        user_msg = (
            f"The user asked: {user_prompt!r}\n\n"
            f"This was translated to the CycL query:\n  {failed_query}\n"
            f"  variable: {query_var}\n\n"
            f"OpenCyc returned NO bindings for this query.\n\n"
            f"Please supply factual information to fill the gap."
        )
        sentences, direct_assertions = self._call_ollama_dual(
            system=_LLM_AUG_GENERAL_SYSTEM, user=user_msg, exec_log=exec_log,
        )

        total = 0
        if sentences:
            pipeline_assertions = self._translate_sentences_to_assertions(
                sentences, entity_hint=None, exec_log=exec_log,
            )
            if pipeline_assertions:
                total += self._assert_facts(pipeline_assertions, source="llm-pipeline", exec_log=exec_log)
        if direct_assertions:
            total += self._assert_facts(direct_assertions, source="llm-direct", exec_log=exec_log)

        return total > 0

    # =============================
    # Execution + repairs
    # =============================

    def _execute_query_with_repairs(
        self, *, query_type: str, query: str, var: str, limit: int,
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
                    bindings = self.cyc.ask_var(mt=self.session_info.session_mt, query=query, var=var, limit=int(limit))
                    exec_log.append({"type": "ask_var", "mt": self.session_info.session_mt, "query": query, "var": var, "limit": int(limit), "bindings": bindings})
                    return bindings
                raise RuntimeError(f"Unknown query_type: {query_type}")
            except CycBridgeError as e:
                msg = e.server_message
                exec_log.append({"type": "cyc_error", "message": msg})
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
                        continue
                    if exists:
                        continue
                    if self._safe_to_autocreate_constant(tok):
                        safe_missing.append(tok)
                if not safe_missing:
                    raise
                repairs += 1
                if repairs > int(self.settings.max_missing_term_repairs):
                    raise RuntimeError(f"Exceeded MAX_MISSING_TERM_REPAIRS={self.settings.max_missing_term_repairs}. Last error: {msg}") from e
                self._progress("defining missing constants in session MT...")
                self._define_missing_constants(safe_missing, query=query, exec_log=exec_log)

    def _safe_to_autocreate_constant(self, const_tok: str) -> bool:
        if not const_tok.startswith("#$") or len(const_tok) < 3:
            return False
        return const_tok[2].isupper() or const_tok[2].isdigit()

    def _define_missing_constants(self, missing_constants: Sequence[str], *, query: str, exec_log: List[Dict[str, Any]]) -> None:
        seen: Set[str] = set()
        for tok in missing_constants:
            if tok in seen:
                continue
            seen.add(tok)
            bare = tok[2:] if tok.startswith("#$") else tok
            if not bare:
                continue
            try:
                created_name = self.cyc.create_constant(bare)
                exec_log.append({"type": "ensure_term", "name": bare, "created": created_name})
            except CycBridgeError as e:
                exec_log.append({"type": "ensure_term", "name": bare, "error": e.server_message})
                continue
            try:
                arities = _find_head_arities(query, tok)
                if arities:
                    arity = max(arities)
                    pred_type = {1: "#$UnaryPredicate", 2: "#$BinaryPredicate", 3: "#$TernaryPredicate", 4: "#$QuaternaryPredicate"}.get(arity, "#$Predicate")
                    sent = f"(#$isa {tok} {pred_type})"
                else:
                    sent = f"(#$isa {tok} #$Thing)"
                self.cyc.assert_sentence(mt=self.session_info.session_mt, sentence=sent)
                exec_log.append({"type": "assert", "mt": self.session_info.session_mt, "sentence": sent})
            except CycBridgeError as e:
                exec_log.append({"type": "assert", "mt": self.session_info.session_mt, "sentence": sent, "error": e.server_message})

    # =============================
    # Natural-language rendering
    # =============================

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
        self, *, query_type: str, query: str, var: str, raw_value: Any,
    ) -> Tuple[str, List[str], List[str]]:
        evidence: List[str] = []
        limitations: List[str] = []
        evidence.append(f"{query_type}: {query}")
        if query_type == "ask_true":
            return ("true" if bool(raw_value) else "false"), evidence, limitations
        bindings = raw_value if isinstance(raw_value, list) else []
        if not bindings:
            limitations.append("OpenCyc returned no bindings for the translated query.")
            return "(no bindings)", evidence, limitations
        rendered = [self._term_to_english(str(b)) for b in bindings]
        if self.debug:
            evidence.append(f"var={var}  count={len(bindings)}")
        return "\n".join(rendered), evidence, limitations
