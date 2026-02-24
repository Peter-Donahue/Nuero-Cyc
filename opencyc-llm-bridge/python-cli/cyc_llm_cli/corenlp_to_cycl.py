"""
corenlp_to_cycl.py

CoreNLP-backed rule-based English -> CycL converter.

This module is intended to be used as a structured intermediate representation (IR) layer:
CoreNLP produces tokenization, lemmas, NER, and dependency parses, and this module maps
those structures into a constrained CycL fragment.

Key features (as used by the CLI orchestrator):
- CoreNLP REST client: CoreNLPServerClient
- Cyc bridge REST client: CycBridgeClient (for lexicon lookup/scoring)
- Cyc lexicon mapping: CycLexicon
- Candidate ranking: CycCandidateScorer (cheap isa/genls/arity/nameString checks)
- Translator: CycLTranslator (dependency-driven composition + quantifiers)

Notes
-----
- This is not CycNL. Without a strong Cyc lexicon/ontology mapping layer, output will often
  contain fallback constants (e.g. #$Paint, #$Dog) that may not exist in your KB.
- The translator intentionally aims to generate CycL that is easy for OpenCyc to accept:
  single fully parenthesized sentences, with explicit #$isa restrictors and simple connectives.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


# ----------------------------
# Module-level configuration
# ----------------------------

_DEFAULT_CORENLP_URL = "http://localhost:9000"
_DEFAULT_CYC_BRIDGE_URL = "http://localhost:8081"
_DEFAULT_TIMEOUT_SEC = 60

_DEFAULT_ANNOTATORS = "tokenize,ssplit,pos,lemma,ner,depparse"
_DEFAULT_PROPERTIES: Dict[str, Any] = {
    "annotators": _DEFAULT_ANNOTATORS,
    "outputFormat": "json",
}

_DEFAULT_CYC_LEXICON_MT = "#$EnglishMt"
_DEFAULT_CYC_QUERY_MT = "#$BaseKB"
_DEFAULT_CYC_LEX_LIMIT = 25


# Optional Ollama-controlled mappings (enable with USE_OLLAMA_MAPPINGS=1)
_DEFAULT_USE_OLLAMA_MAPPINGS = (os.getenv("USE_OLLAMA_MAPPINGS") or "").strip().lower() in ("1", "true", "t", "yes", "y", "on")
_DEFAULT_OLLAMA_BASE_URL = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
_DEFAULT_OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "llama3")
try:
    _DEFAULT_OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE") or "0.0")
except ValueError:
    _DEFAULT_OLLAMA_TEMPERATURE = 0.0
try:
    _DEFAULT_OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC") or str(_DEFAULT_TIMEOUT_SEC))
except ValueError:
    _DEFAULT_OLLAMA_TIMEOUT_SEC = _DEFAULT_TIMEOUT_SEC

# Determiners (intentionally conservative)
_EXISTENTIAL_DETERMINERS = {"a", "an", "the", "some"}
_UNIVERSAL_DETERMINERS = {"every", "all", "each", "forall"}
_NEGATIVE_DETERMINERS = {"no", "none", "neither"}

# WH words used as query variables
_WH_WORDS_TO_VAR = {
    "who": "?Who",
    "whom": "?Who",
    "what": "?What",
    "which": "?Which",
    "where": "?Where",
    "when": "?When",
}

# Pronoun mapping
_PRONOUN_TO_VAR = {
    "i": "?Speaker",
    "me": "?Speaker",
    "my": "?Speaker",
    "mine": "?Speaker",
    "you": "?TargetAgent",
    "your": "?TargetAgent",
    "yours": "?TargetAgent",
}

# Small demo fallback lexicon (extend/replace via CycLexicon bridge)
_NOUN_LEMMA_TO_CYC_COLLECTION: Dict[str, str] = {
    "man": "#$AdultMalePerson",
    "woman": "#$AdultFemalePerson",
    "person": "#$Person",
    "human": "#$Person",
    "dog": "#$Dog",
    "cat": "#$Cat",
}

_VERB_LEMMA_TO_CYC_PREDICATE: Dict[str, str] = {
    "like": "#$likesObject",
    "admire": "#$admire",
    "paint": "#$paints",
}

_ADJ_LEMMA_TO_CYC_TRAIT: Dict[str, str] = {
    "red": "#$RedColor",
    "blue": "#$BlueColor",
    "green": "#$GreenColor",
}

# Dependency labels (CoreNLP mixes UD/SD; enhanced deps may add subtype suffixes)
_SUBJECT_DEPS = ("nsubj", "nsubj:pass", "csubj", "csubj:pass")
_OBJECT_DEPS = ("obj", "dobj", "iobj")
_COPULA_DEPS = ("cop",)
_DET_DEPS_PREFIX = "det"
_AMOD_DEPS = ("amod",)
_NEG_DEPS = ("neg",)

_RELCL_DEPS_PREFIX = "acl:relcl"
_RELCL_DEPS_FALLBACK = ("rcmod", "acl")

# For multiword names/compounds
_NAME_RELATIONS = ("compound", "flat", "name")

# NER -> preferred #$isa for proper terms (best-effort; adjust for your KB)
_CYC_NER_TO_PREFERRED_ISA: Dict[str, str] = {
    "PERSON": "#$Person",
    "ORGANIZATION": "#$Organization",
    "ORG": "#$Organization",
    "LOCATION": "#$SpatialThing",
    "CITY": "#$City",
    "STATE_OR_PROVINCE": "#$GeopoliticalEntity",
    "COUNTRY": "#$GeographicalRegion",
}


# ----------------------------
# Optional controlled LLM mappings (Ollama)
# ----------------------------

_CYC_CONST_RE = re.compile(r"^#\$(?:[A-Za-z][A-Za-z0-9_-]*)$")
_CYC_VAR_RE = re.compile(r"^\?[A-Za-z][A-Za-z0-9_-]*$")


def _is_cyc_constant_name(text: str) -> bool:
    return bool(_CYC_CONST_RE.match((text or "").strip()))


def _normalize_cyc_constant_name(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if t.startswith("#$") and _is_cyc_constant_name(t):
        return t
    if _is_cyc_constant_name("#$" + t):
        return "#$" + t
    return ""


class MappingProvider:
    def determiner_type(self, det: str) -> str:
        raise NotImplementedError

    def wh_info(self, word: str) -> Tuple[str, str]:
        raise NotImplementedError

    def pronoun_var(self, word: str) -> str:
        raise NotImplementedError

    def fallback_noun_collections(self, lemma: str) -> List[str]:
        raise NotImplementedError

    def fallback_verb_predicates(self, lemma: str) -> List[str]:
        raise NotImplementedError

    def fallback_adj_traits(self, lemma: str) -> List[str]:
        raise NotImplementedError

    def ner_preferred_isa(self, ner: str) -> str:
        raise NotImplementedError


class StaticMappingProvider(MappingProvider):
    def determiner_type(self, det: str) -> str:
        d = (det or "").strip().lower()
        if not d:
            return "other"
        if d in _NEGATIVE_DETERMINERS:
            return "negative"
        if d in _UNIVERSAL_DETERMINERS:
            return "universal"
        if d in _EXISTENTIAL_DETERMINERS:
            return "existential"
        return "other"

    def wh_info(self, word: str) -> Tuple[str, str]:
        w = (word or "").strip().lower()
        v = _WH_WORDS_TO_VAR.get(w, "")
        if not v:
            return "", ""
        if w in ("who", "whom"):
            return v, "person"
        if w == "where":
            return v, "place"
        if w == "when":
            return v, "time"
        if w == "which":
            return v, "which"
        return v, "thing"

    def pronoun_var(self, word: str) -> str:
        return _PRONOUN_TO_VAR.get((word or "").strip().lower(), "")

    def fallback_noun_collections(self, lemma: str) -> List[str]:
        l = (lemma or "").strip().lower()
        v = _NOUN_LEMMA_TO_CYC_COLLECTION.get(l, "")
        return [v] if v else []

    def fallback_verb_predicates(self, lemma: str) -> List[str]:
        l = (lemma or "").strip().lower()
        v = _VERB_LEMMA_TO_CYC_PREDICATE.get(l, "")
        return [v] if v else []

    def fallback_adj_traits(self, lemma: str) -> List[str]:
        l = (lemma or "").strip().lower()
        v = _ADJ_LEMMA_TO_CYC_TRAIT.get(l, "")
        return [v] if v else []

    def ner_preferred_isa(self, ner: str) -> str:
        return _CYC_NER_TO_PREFERRED_ISA.get((ner or "").strip().upper(), "")


class OllamaClientError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, *, timeout_sec: int = _DEFAULT_OLLAMA_TIMEOUT_SEC):
        self._base_url = (base_url or "").rstrip("/")
        self._timeout_sec = int(timeout_sec)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise OllamaClientError(f"Ollama HTTP {e.code} at {path}: {body[:500]}") from e
        except urllib.error.URLError as e:
            raise OllamaClientError(f"Ollama unreachable at {self._base_url}: {e}") from e

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise OllamaClientError(f"Ollama returned invalid JSON: {body[:500]}") from e

    def chat_json(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if schema is not None:
            payload["format"] = schema
        else:
            payload["format"] = "json"
        if options is not None:
            payload["options"] = options
        return self._post_json("/api/chat", payload)


def _extract_json_from_ollama_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    msg = (resp or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    s = content.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # best-effort: extract first {...} object
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}


class OllamaMappingProvider(MappingProvider):
    """Controlled, cached mappings via Ollama structured outputs.

    This is intentionally constrained:
      - Each call is a small classification/mapping problem.
      - Output must match a tight JSON schema.
      - Results are validated and sanitized before use.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_OLLAMA_BASE_URL,
        model: str = _DEFAULT_OLLAMA_MODEL,
        temperature: float = _DEFAULT_OLLAMA_TEMPERATURE,
        timeout_sec: int = _DEFAULT_OLLAMA_TIMEOUT_SEC,
        bridge: Optional["CycBridgeClient"] = None,
        fallback: Optional[MappingProvider] = None,
    ):
        self._client = OllamaClient(base_url, timeout_sec=int(timeout_sec))
        self._model = model
        self._temperature = float(temperature)
        self._bridge = bridge
        self._fallback = fallback or StaticMappingProvider()

        self._det_cache: Dict[str, str] = {}
        self._wh_cache: Dict[str, Tuple[str, str]] = {}
        self._pron_cache: Dict[str, str] = {}
        self._noun_cache: Dict[str, List[str]] = {}
        self._verb_cache: Dict[str, List[str]] = {}
        self._adj_cache: Dict[str, List[str]] = {}
        self._ner_cache: Dict[str, str] = {}

    def _options(self) -> Dict[str, Any]:
        return {
            "temperature": self._temperature,
            "num_predict": 256,
        }

    def _call(self, *, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            resp = self._client.chat_json(model=self._model, messages=messages, schema=schema, options=self._options())
        except Exception:
            # If schema mode fails (older Ollama), retry with plain json mode.
            try:
                resp = self._client.chat_json(model=self._model, messages=messages, schema=None, options=self._options())
            except Exception:
                return {}
        return _extract_json_from_ollama_response(resp)

    def determiner_type(self, det: str) -> str:
        d = (det or "").strip().lower()
        if not d:
            return "other"
        if d in self._det_cache:
            return self._det_cache[d]

        schema = {
            "type": "object",
            "properties": {
                "det_type": {"type": "string", "enum": ["existential", "universal", "negative", "other"]},
            },
            "required": ["det_type"],
            "additionalProperties": False,
        }
        system = "Return only JSON. No explanation."
        user = (
            "Classify the English determiner token into one of: "
            "existential, universal, negative, other.\n"
            f"token: {d!r}\n"
            "Guidance: existential includes a/an/the/some; universal includes every/all/each/forall; "
            "negative includes no/none/neither."
        )
        data = self._call(system=system, user=user, schema=schema)
        det_type = str(data.get("det_type", "other")).strip().lower()
        if det_type not in ("existential", "universal", "negative", "other"):
            det_type = "other"
        if det_type == "other":
            # allow fallback to static lists for determiners
            det_type = self._fallback.determiner_type(d)
        self._det_cache[d] = det_type
        return det_type

    def wh_info(self, word: str) -> Tuple[str, str]:
        w = (word or "").strip().lower()
        if not w:
            return "", ""
        if w in self._wh_cache:
            return self._wh_cache[w]

        schema = {
            "type": "object",
            "properties": {
                "var": {"type": ["string", "null"]},
                "kind": {"type": ["string", "null"]},
            },
            "required": ["var", "kind"],
            "additionalProperties": False,
        }
        system = "Return only JSON. No explanation."
        user = (
            "Map a single English wh-word to a CycL query variable.\n"
            "Valid vars: ?Who, ?What, ?Which, ?Where, ?When, or null.\n"
            "Valid kinds: person, thing, which, place, time, or null.\n"
            f"token: {w!r}"
        )
        data = self._call(system=system, user=user, schema=schema)
        var = (data.get("var") or "").strip()
        kind = (data.get("kind") or "").strip().lower()
        if var not in ("?Who", "?What", "?Which", "?Where", "?When"):
            var = ""
        if kind not in ("person", "thing", "which", "place", "time"):
            kind = ""
        if not var:
            var, kind = self._fallback.wh_info(w)
        self._wh_cache[w] = (var, kind)
        return var, kind

    def pronoun_var(self, word: str) -> str:
        w = (word or "").strip().lower()
        if not w:
            return ""
        if w in self._pron_cache:
            return self._pron_cache[w]

        schema = {
            "type": "object",
            "properties": {"var": {"type": ["string", "null"]}},
            "required": ["var"],
            "additionalProperties": False,
        }
        system = "Return only JSON. No explanation."
        user = (
            "Map the English pronoun token to a CycL variable.\n"
            "Use ?Speaker for first-person singular (i, me, my, mine).\n"
            "Use ?TargetAgent for second-person (you, your, yours).\n"
            "Otherwise return null.\n"
            f"token: {w!r}"
        )
        data = self._call(system=system, user=user, schema=schema)
        var = (data.get("var") or "").strip()
        if var not in ("?Speaker", "?TargetAgent"):
            var = ""
        if not var:
            var = self._fallback.pronoun_var(w)
        self._pron_cache[w] = var
        return var

    def _filter_existing_constants(self, cands: List[str]) -> List[str]:
        if self._bridge is None:
            return cands
        existing: List[str] = []
        checked = 0
        for c in cands:
            try:
                if self._bridge.constant_exists(name=c):
                    existing.append(c)
                checked += 1
            except Exception:
                continue
        if existing:
            return existing
        # If we successfully checked existence and none exist, prefer to return no candidates
        # (fall back to deterministic constantization instead of using hallucinated constants).
        if checked > 0:
            return []
        return cands

    def fallback_noun_collections(self, lemma: str) -> List[str]:
        l = (lemma or "").strip().lower()
        if not l:
            return []
        if l in self._noun_cache:
            return self._noun_cache[l]

        schema = {
            "type": "object",
            "properties": {
                "candidates": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }
        system = "Return only JSON. No explanation."
        user = (
            "Suggest OpenCyc collection constants (CycL constants) for an English common noun lemma.\n"
            "Return up to 5 candidate constants in decreasing likelihood.\n"
            "Each candidate must be a single Cyc constant name like #$AdultMalePerson.\n"
            "If unsure, return an empty list.\n"
            f"lemma: {l!r}"
        )
        data = self._call(system=system, user=user, schema=schema)
        raw = data.get("candidates") or []
        cands: List[str] = []
        if isinstance(raw, list):
            for x in raw:
                if not isinstance(x, str):
                    continue
                c = _normalize_cyc_constant_name(x)
                if c and c not in cands:
                    cands.append(c)
        if not cands:
            cands = self._fallback.fallback_noun_collections(l)
        cands = self._filter_existing_constants(cands)
        self._noun_cache[l] = cands
        return cands

    def fallback_verb_predicates(self, lemma: str) -> List[str]:
        l = (lemma or "").strip().lower()
        if not l:
            return []
        if l in self._verb_cache:
            return self._verb_cache[l]

        schema = {
            "type": "object",
            "properties": {
                "candidates": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }
        system = "Return only JSON. No explanation."
        user = (
            "Suggest OpenCyc predicate constants for an English verb lemma.\n"
            "Return up to 5 candidate predicate constant names like #$likesObject.\n"
            "If unsure, return an empty list.\n"
            f"lemma: {l!r}"
        )
        data = self._call(system=system, user=user, schema=schema)
        raw = data.get("candidates") or []
        cands: List[str] = []
        if isinstance(raw, list):
            for x in raw:
                if not isinstance(x, str):
                    continue
                c = _normalize_cyc_constant_name(x)
                if c and c not in cands:
                    cands.append(c)
        if not cands:
            cands = self._fallback.fallback_verb_predicates(l)
        cands = self._filter_existing_constants(cands)
        self._verb_cache[l] = cands
        return cands

    def fallback_adj_traits(self, lemma: str) -> List[str]:
        l = (lemma or "").strip().lower()
        if not l:
            return []
        if l in self._adj_cache:
            return self._adj_cache[l]

        schema = {
            "type": "object",
            "properties": {
                "candidates": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }
        system = "Return only JSON. No explanation."
        user = (
            "Suggest OpenCyc constants suitable as traits/collections for adjectives, to be used as the third argument to #$hasAttributeOrCollection.\n"
            "Return up to 5 candidate constant names like #$RedColor.\n"
            "If unsure, return an empty list.\n"
            f"lemma: {l!r}"
        )
        data = self._call(system=system, user=user, schema=schema)
        raw = data.get("candidates") or []
        cands: List[str] = []
        if isinstance(raw, list):
            for x in raw:
                if not isinstance(x, str):
                    continue
                c = _normalize_cyc_constant_name(x)
                if c and c not in cands:
                    cands.append(c)
        if not cands:
            cands = self._fallback.fallback_adj_traits(l)
        cands = self._filter_existing_constants(cands)
        self._adj_cache[l] = cands
        return cands

    def ner_preferred_isa(self, ner: str) -> str:
        n = (ner or "").strip().upper()
        if not n:
            return ""
        if n in self._ner_cache:
            return self._ner_cache[n]

        schema = {
            "type": "object",
            "properties": {"isa": {"type": ["string", "null"]}},
            "required": ["isa"],
            "additionalProperties": False,
        }
        system = "Return only JSON. No explanation."
        user = (
            "Map a CoreNLP NER tag to a preferred OpenCyc collection constant used with #$isa.\n"
            "Return a Cyc constant like #$Person, or null if none.\n"
            f"ner: {n!r}\n"
            "Typical: PERSON->#$Person, ORGANIZATION/ORG->#$Organization, LOCATION->#$SpatialThing, CITY->#$City, "
            "STATE_OR_PROVINCE->#$GeopoliticalEntity, COUNTRY->#$GeographicalRegion."
        )
        data = self._call(system=system, user=user, schema=schema)
        isa_raw = data.get("isa")
        isa = _normalize_cyc_constant_name(isa_raw) if isinstance(isa_raw, str) else ""
        if not isa:
            isa = self._fallback.ner_preferred_isa(n)
        self._ner_cache[n] = isa
        return isa


def _default_mapping_provider(*, bridge: Optional["CycBridgeClient"]) -> MappingProvider:
    if _DEFAULT_USE_OLLAMA_MAPPINGS:
        return OllamaMappingProvider(
            base_url=_DEFAULT_OLLAMA_BASE_URL,
            model=_DEFAULT_OLLAMA_MODEL,
            temperature=_DEFAULT_OLLAMA_TEMPERATURE,
            timeout_sec=_DEFAULT_OLLAMA_TIMEOUT_SEC,
            bridge=bridge,
        )
    return StaticMappingProvider()



# ----------------------------
# CycL AST representation
# ----------------------------

CycLTerm = Union[str, List["CycLTerm"]]


def _cycl(fun: str, *args: CycLTerm) -> CycLTerm:
    return [fun, *args]


def _flatten_and(args: Sequence[CycLTerm]) -> List[CycLTerm]:
    flat: List[CycLTerm] = []
    for a in args:
        if isinstance(a, list) and a and a[0] == "#$and":
            flat.extend(_flatten_and(a[1:]))
        else:
            flat.append(a)
    return flat


def _and(*args: CycLTerm) -> CycLTerm:
    conjuncts = [a for a in _flatten_and(args) if a is not None]
    if not conjuncts:
        return "#$True"
    if len(conjuncts) == 1:
        return conjuncts[0]
    return ["#$and", *conjuncts]


def _or(*args: CycLTerm) -> CycLTerm:
    disj = [a for a in args if a is not None]
    if not disj:
        return "#$False"
    if len(disj) == 1:
        return disj[0]
    return ["#$or", *disj]


def _not(a: CycLTerm) -> CycLTerm:
    return ["#$not", a]


def _implies(a: CycLTerm, b: CycLTerm) -> CycLTerm:
    return ["#$implies", a, b]


def _there_exists(var: str, body: CycLTerm) -> CycLTerm:
    return ["#$thereExists", var, body]


def _for_all(var: str, body: CycLTerm) -> CycLTerm:
    return ["#$forAll", var, body]


def cycl_to_string(term: CycLTerm) -> str:
    if isinstance(term, str):
        return term
    if not term:
        return "()"
    return "(" + " ".join(cycl_to_string(x) for x in term) + ")"


# ----------------------------
# CoreNLP server client (REST)
# ----------------------------

class CoreNLPServerClient:
    """Minimal CoreNLP server client (JSON output)."""

    def __init__(self, base_url: str = _DEFAULT_CORENLP_URL, timeout_sec: int = _DEFAULT_TIMEOUT_SEC):
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = int(timeout_sec)

    def annotate(self, text: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        props = properties or _DEFAULT_PROPERTIES
        props_str = json.dumps(props)

        body = urllib.parse.urlencode({"data": text}).encode("utf-8")
        query = urllib.parse.urlencode({"properties": props_str})
        url = f"{self._base_url}/?{query}"

        req = urllib.request.Request(url=url, data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
        with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload)


# ----------------------------
# Cyc bridge client (REST)
# ----------------------------

class CycBridgeClient:
    """HTTP client for the Java CycBridgeServer.

    This client is used by the translator for lexicon mapping/scoring. The CLI orchestrator
    uses its own bridge client for session/query execution.
    """

    def __init__(self, base_url: str = _DEFAULT_CYC_BRIDGE_URL, timeout_sec: int = _DEFAULT_TIMEOUT_SEC):
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = int(timeout_sec)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CycBridge HTTP {e.code} at {path}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"CycBridge unreachable at {self._base_url}: {e}") from e

    def ask_true(self, *, query: str, mt: str = _DEFAULT_CYC_QUERY_MT) -> bool:
        resp = self._post_json("/api/v1/ask_true", {"mt": mt, "query": query})
        return bool(resp.get("answer", False))

    def ask_var(self, *, query: str, var: str = "?X", mt: str = _DEFAULT_CYC_QUERY_MT, limit: int = 50) -> List[str]:
        resp = self._post_json("/api/v1/ask_var", {"mt": mt, "query": query, "var": var, "limit": int(limit)})
        bindings = resp.get("bindings", []) or []
        return [str(x) for x in bindings]

    def constant_exists(self, *, name: str) -> bool:
        resp = self._post_json("/api/v1/constant/exists", {"name": name})
        return bool(resp.get("exists", False))

    def converse(self, *, subl: str) -> str:
        resp = self._post_json("/api/v1/converse", {"subl": subl})
        return str(resp.get("result", ""))


def _escape_cyc_string(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace("\"", "\\\"")


def _cyc_word_list(text: str) -> str:
    toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    if not toks:
        toks = [str(text or "").lower()]
    inner = " ".join(f'"{_escape_cyc_string(t)}"' for t in toks)
    return f"({inner})"


# ----------------------------
# Lexicon / ontology mapping
# ----------------------------

@dataclass(frozen=True)
class VerbLexiconEntry:
    template: Optional[CycLTerm]
    predicate: Optional[str]
    raw: Optional[str] = None


class CycLexicon:
    """Cyc-backed lexicon mapping (candidate generation only)."""

    def __init__(
        self,
        bridge: Optional[CycBridgeClient] = None,
        lex_mt: str = _DEFAULT_CYC_LEXICON_MT,
        lex_limit: int = _DEFAULT_CYC_LEX_LIMIT,
        mapper: Optional[MappingProvider] = None,
    ):
        self._bridge = bridge
        self._lex_mt = lex_mt
        self._lex_limit = int(lex_limit)
        self._mapper: MappingProvider = mapper or StaticMappingProvider()

        self._noun_cache: Dict[str, List[str]] = {}
        self._verb_cache: Dict[str, Tuple[List[str], List[str]]] = {}
        self._adj_cache: Dict[str, Tuple[List[str], List[str]]] = {}
        self._proper_cache: Dict[str, List[str]] = {}

    @property
    def bridge(self) -> Optional[CycBridgeClient]:
        return self._bridge

    @property
    def lex_mt(self) -> str:
        return self._lex_mt

    @property
    def lex_limit(self) -> int:
        return self._lex_limit

    @property
    def mapper(self) -> MappingProvider:
        return self._mapper

    def set_mapper(self, mapper: MappingProvider) -> None:
        self._mapper = mapper
        # Clear caches so fallback candidates refresh.
        self._noun_cache.clear()
        self._verb_cache.clear()
        self._adj_cache.clear()

    def noun_candidates(self, lemma: str) -> List[str]:
        key = (lemma or "").lower()
        if key in self._noun_cache:
            return self._noun_cache[key]

        out: List[str] = []
        if self._bridge is not None and key:
            wl = _cyc_word_list(key)
            noun_pos = ["#$SimpleNoun", "#$MassNoun", "#$AgentiveNoun", "#$Noun"]
            for pos in noun_pos:
                for q in (
                    f"(#$and (#$lex ?FORM ?W {wl}) (#$denotation ?W {pos} 0 ?DENOT))",
                    f"(#$and (#$lex ?FORM ?W {wl}) (#$denotationRelatedTo ?W {pos} 0 ?DENOT))",
                ):
                    try:
                        out.extend(self._bridge.ask_var(query=q, var="?DENOT", mt=self._lex_mt, limit=self._lex_limit))
                    except Exception:
                        continue

        if not out:
            out.extend(self._mapper.fallback_noun_collections(key))

        out = list(dict.fromkeys(out))
        self._noun_cache[key] = out
        return out

    def verb_candidates(self, lemma: str) -> Tuple[List[str], List[str]]:
        key = (lemma or "").lower()
        if key in self._verb_cache:
            return self._verb_cache[key]

        denots: List[str] = []
        templates: List[str] = []

        if self._bridge is not None and key:
            wl = _cyc_word_list(key)
            qt = f"(#$and (#$lex ?FORM ?W {wl}) (#$verbSemTrans ?W ?IDX ?FRAME ?TEMPLATE))"
            qd = f"(#$and (#$lex ?FORM ?W {wl}) (#$denotation ?W #$Verb 0 ?DENOT))"
            try:
                templates.extend(self._bridge.ask_var(query=qt, var="?TEMPLATE", mt=self._lex_mt, limit=self._lex_limit))
            except Exception:
                pass
            try:
                denots.extend(self._bridge.ask_var(query=qd, var="?DENOT", mt=self._lex_mt, limit=self._lex_limit))
            except Exception:
                pass

        if not denots:
            denots.extend(self._mapper.fallback_verb_predicates(key))

        denots = list(dict.fromkeys(denots))
        templates = list(dict.fromkeys(templates))
        self._verb_cache[key] = (denots, templates)
        return denots, templates

    def adj_candidates(self, lemma: str) -> Tuple[List[str], List[str]]:
        key = (lemma or "").lower()
        if key in self._adj_cache:
            return self._adj_cache[key]

        traits: List[str] = []
        templates: List[str] = []

        if self._bridge is not None and key:
            wl = _cyc_word_list(key)
            q1 = f"(#$and (#$lex ?FORM ?W {wl}) (#$denotation ?W #$Adjective 0 ?DENOT))"
            q2 = f"(#$and (#$lex ?FORM ?W {wl}) (#$denotationRelatedTo ?W #$Adjective 0 ?DENOT))"
            qt = f"(#$and (#$lex ?FORM ?W {wl}) (#$adjSemTrans ?W ?IDX ?FRAME ?TEMPLATE))"
            for q in (q1, q2):
                try:
                    traits.extend(self._bridge.ask_var(query=q, var="?DENOT", mt=self._lex_mt, limit=self._lex_limit))
                except Exception:
                    pass
            try:
                templates.extend(self._bridge.ask_var(query=qt, var="?TEMPLATE", mt=self._lex_mt, limit=self._lex_limit))
            except Exception:
                pass

        if not traits:
            traits.extend(self._mapper.fallback_adj_traits(key))

        traits = list(dict.fromkeys(traits))
        templates = list(dict.fromkeys(templates))
        self._adj_cache[key] = (traits, templates)
        return traits, templates

    def proper_candidates(self, text: str) -> List[str]:
        key = (text or "").strip()
        if key in self._proper_cache:
            return self._proper_cache[key]

        out: List[str] = []
        if self._bridge is not None and key:
            s = _escape_cyc_string(key)
            preds = ["#$preferredNameString", "#$nameString"]
            for pred in preds:
                q = f"({pred} ?T \"{s}\")"
                try:
                    out.extend(self._bridge.ask_var(query=q, var="?T", mt=self._lex_mt, limit=self._lex_limit))
                except Exception:
                    continue

        out = list(dict.fromkeys(out))
        self._proper_cache[key] = out
        return out


# ----------------------------
# Cyc candidate scoring
# ----------------------------

class CycCandidateScorer:
    """Rank lexicon candidates using cheap Cyc queries."""

    def __init__(self, bridge: CycBridgeClient, *, query_mt: str = _DEFAULT_CYC_QUERY_MT, mapper: Optional[MappingProvider] = None):
        self._bridge = bridge
        self._query_mt = query_mt
        self._mapper: MappingProvider = mapper or StaticMappingProvider()
        self._ask_true_cache: Dict[str, bool] = {}
        self._ask_var_cache: Dict[Tuple[str, str, int], List[str]] = {}

    def _true(self, query: str) -> bool:
        q = query.strip()
        if q in self._ask_true_cache:
            return self._ask_true_cache[q]
        try:
            ans = bool(self._bridge.ask_true(query=q, mt=self._query_mt))
        except Exception:
            ans = False
        self._ask_true_cache[q] = ans
        return ans

    def _var(self, query: str, var: str, limit: int = 10) -> List[str]:
        key = (query.strip(), var, int(limit))
        if key in self._ask_var_cache:
            return self._ask_var_cache[key]
        try:
            vals = self._bridge.ask_var(query=key[0], var=var, mt=self._query_mt, limit=int(limit))
        except Exception:
            vals = []
        self._ask_var_cache[key] = vals
        return vals

    def score_noun_collection(self, cand: str, *, lemma: str) -> float:
        c = (cand or "").strip()
        if not c:
            return -1e9
        score = 0.0
        if self._true(f"(#$isa {c} #$Collection)"):
            score += 10.0
        if self._true(f"(#$isa {c} #$Predicate)"):
            score -= 10.0
        if self._true(f"(#$isa {c} #$Individual)"):
            score -= 5.0
        lemma_l = (lemma or "").lower()
        if lemma_l and lemma_l in c.lower():
            score += 1.0
        return score

    def score_proper_term(self, cand: str, *, text: str, ner: str = "O") -> float:
        c = (cand or "").strip()
        if not c:
            return -1e9
        score = 0.0
        if self._true(f"(#$isa {c} #$Individual)"):
            score += 10.0
        if self._true(f"(#$isa {c} #$Collection)"):
            score -= 5.0

        pref_isa = self._mapper.ner_preferred_isa(ner)
        if pref_isa and self._true(f"(#$isa {c} {pref_isa})"):
            score += 3.0

        s = _escape_cyc_string(text or "")
        if s:
            if self._true(f"(#$preferredNameString {c} \"{s}\")"):
                score += 4.0
            elif self._true(f"(#$nameString {c} \"{s}\")"):
                score += 2.0

        return score

    def score_verb_predicate(
        self,
        cand: str,
        *,
        lemma: str,
        transitive: bool,
        subj_type: Optional[str] = None,
        obj_type: Optional[str] = None,
    ) -> float:
        c = (cand or "").strip()
        if not c:
            return -1e9
        score = 0.0

        if self._true(f"(#$isa {c} #$Predicate)"):
            score += 10.0
        else:
            score -= 10.0

        if transitive:
            if self._true(f"(#$isa {c} #$BinaryPredicate)"):
                score += 4.0
            if self._true(f"(#$arity {c} 2)"):
                score += 2.0
        else:
            if self._true(f"(#$isa {c} #$UnaryPredicate)"):
                score += 4.0
            if self._true(f"(#$arity {c} 1)"):
                score += 2.0

        lemma_l = (lemma or "").lower()
        if lemma_l and lemma_l in c.lower():
            score += 1.0

        if subj_type:
            if self._true(f"(#$arg1Isa {c} {subj_type})"):
                score += 2.0
            else:
                for t in self._var(f"(#$arg1Isa {c} ?T)", var="?T", limit=5):
                    if self._true(f"(#$genls {subj_type} {t})"):
                        score += 1.0
                        break

        if transitive and obj_type:
            if self._true(f"(#$arg2Isa {c} {obj_type})"):
                score += 2.0
            else:
                for t in self._var(f"(#$arg2Isa {c} ?T)", var="?T", limit=5):
                    if self._true(f"(#$genls {obj_type} {t})"):
                        score += 1.0
                        break

        return score


# ----------------------------
# CycL template parsing (verbSemTrans/adjSemTrans)
# ----------------------------

_TOKEN_RE = re.compile(r"\s*(?:(\()|(\))|(\"(?:\\.|[^\"\\])*\")|([^\s()]+))", re.DOTALL)


def _strip_cycl_comments(text: str) -> str:
    s = text or ""
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    s = re.sub(r";[^\n]*", " ", s)
    s = re.sub(r"//[^\n]*", " ", s)
    return s


def parse_cycl_sexp(text: str) -> CycLTerm:
    src = _strip_cycl_comments(text).strip()
    if not src:
        raise ValueError("Empty CycL string")

    tokens: List[str] = []
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise ValueError(f"Failed to tokenize CycL at: {src[i:i+50]!r}")
        i = m.end()
        if m.group(1):
            tokens.append("(")
        elif m.group(2):
            tokens.append(")")
        elif m.group(3):
            tokens.append(m.group(3))
        elif m.group(4):
            tokens.append(m.group(4))

    pos = 0

    def parse_one() -> CycLTerm:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected EOF")
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            lst: List[CycLTerm] = []
            while True:
                if pos >= len(tokens):
                    raise ValueError("Unclosed '('")
                if tokens[pos] == ")":
                    pos += 1
                    return lst
                lst.append(parse_one())
        if tok == ")":
            raise ValueError("Unexpected ')'")
        pos += 1
        return tok

    term = parse_one()
    if pos != len(tokens):
        extra = " ".join(tokens[pos:pos+10])
        raise ValueError(f"Trailing tokens after CycL parse: {extra}")
    return term


def _substitute_placeholders(term: CycLTerm, mapping: Dict[str, str]) -> Tuple[CycLTerm, Set[str]]:
    used: Set[str] = set()
    if isinstance(term, str):
        if term.startswith(":") and term in mapping:
            used.add(term)
            return mapping[term], used
        return term, used
    out_list: List[CycLTerm] = []
    for t in term:
        new_t, used_t = _substitute_placeholders(t, mapping)
        used |= used_t
        out_list.append(new_t)
    return out_list, used


def _contains_unresolved_placeholder(term: CycLTerm) -> bool:
    if isinstance(term, str):
        return term.startswith(":")
    return any(_contains_unresolved_placeholder(t) for t in term)


def _subst_vars(term: CycLTerm, mapping: Dict[str, str]) -> CycLTerm:
    """Substitute ``?Var`` variables in *term* according to *mapping*."""
    if isinstance(term, str):
        return mapping.get(term, term)
    return [_subst_vars(sub, mapping) for sub in term]


def _extract_isa_collections_rec(term: CycLTerm, var: str, out: List[str]) -> None:
    """Recursively find ``(#$isa <var> <Collection>)`` in *term* and collect the collection constants."""
    if isinstance(term, str):
        return
    if not term:
        return
    # Direct match: (#$isa ?X #$SomeCollection)
    if (
        len(term) == 3
        and term[0] == "#$isa"
        and term[1] == var
        and isinstance(term[2], str)
        and term[2].startswith("#$")
    ):
        out.append(term[2])
        return
    # Recurse into conjunctions and nested formulas.
    for sub in term:
        if isinstance(sub, list):
            _extract_isa_collections_rec(sub, var, out)


# ----------------------------
# Dependency graph utilities
# ----------------------------

@dataclass(frozen=True)
class _Token:
    index: int
    word: str
    lemma: str
    pos: str
    ner: str

    @staticmethod
    def from_json(tok: Dict[str, Any]) -> "_Token":
        return _Token(
            index=int(tok.get("index", 0)),
            word=str(tok.get("word", "")),
            lemma=str(tok.get("lemma", tok.get("word", ""))),
            pos=str(tok.get("pos", "")),
            ner=str(tok.get("ner", "O")),
        )


@dataclass(frozen=True)
class _Dep:
    rel: str
    gov: int
    dep: int

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "_Dep":
        return _Dep(
            rel=str(d.get("dep", "")).lower(),
            gov=int(d.get("governor", 0)),
            dep=int(d.get("dependent", 0)),
        )


class _DepGraph:
    def __init__(self, tokens: Dict[int, _Token], deps: List[_Dep]):
        self._tokens = tokens
        self._deps = deps
        children: Dict[int, List[Tuple[str, int]]] = {}
        parents: Dict[int, List[Tuple[str, int]]] = {}
        for e in deps:
            children.setdefault(e.gov, []).append((e.rel, e.dep))
            parents.setdefault(e.dep, []).append((e.rel, e.gov))
        self._children = children
        self._parents = parents

    @staticmethod
    def from_corenlp_sentence(sent: Dict[str, Any]) -> "_DepGraph":
        toks = {_Token.from_json(t).index: _Token.from_json(t) for t in sent.get("tokens", [])}
        dep_list = (
            sent.get("enhancedPlusPlusDependencies")
            or sent.get("enhancedDependencies")
            or sent.get("basicDependencies")
            or []
        )
        deps = [_Dep.from_json(d) for d in dep_list]
        return _DepGraph(toks, deps)

    def token(self, idx: int) -> _Token:
        return self._tokens[idx]

    def children(self, gov_idx: int) -> List[Tuple[str, int]]:
        return list(self._children.get(gov_idx, []))

    def children_with_rel(self, gov_idx: int, rels: Sequence[str]) -> List[Tuple[str, int]]:
        rel_set = set(rels)
        return [(r, d) for (r, d) in self._children.get(gov_idx, []) if r in rel_set]

    def parents(self, dep_idx: int) -> List[Tuple[str, int]]:
        return list(self._parents.get(dep_idx, []))


    def root(self) -> Optional[int]:
        for e in self._deps:
            if e.gov == 0 and e.rel == "root":
                return e.dep
        for e in self._deps:
            if e.rel == "root":
                return e.dep
        for e in self._deps:
            if e.gov == 0:
                return e.dep
        return None


# ----------------------------
# Translation helpers
# ----------------------------

@dataclass
class _Binding:
    var: str
    quant: str  # exists | forall | no
    restrictor: CycLTerm


class _VarAllocator:
    def __init__(self) -> None:
        self._entity_count = 0
        self._event_count = 0

    def new_entity(self) -> str:
        self._entity_count += 1
        return f"?X{self._entity_count}"

    def new_event(self) -> str:
        self._event_count += 1
        return f"?E{self._event_count}"


def _sanitize_for_cyc_constant(text: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", text)
    if not parts:
        return "Thing"
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _as_cyc_constant_from_text(text: str) -> str:
    return "#$" + _sanitize_for_cyc_constant(text)


def _normalize_name_text(text: str) -> str:
    """Normalize a surface name string for Cyc name lookups.

    - Replaces hyphens between letters with spaces (e.g., "Bill-Clinton" -> "Bill Clinton")
    - Collapses multiple whitespace
    """
    t = (text or "").strip()
    if not t:
        return ""
    # Hyphen between letters => space. Avoid touching things like "COVID-19".
    t = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", t)
    t = re.sub(r"(?<=[A-Za-z])_(?=[A-Za-z])", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _text_seems_proper_name(text: str) -> bool:
    """Heuristic: does a surface span look like a proper name?"""
    t = _normalize_name_text(text)
    if not t:
        return False
    parts = [p for p in t.split() if p]
    if not parts:
        return False
    caps = sum(1 for p in parts if p[:1].isupper())
    if caps >= 2:
        return True
    if caps == 1 and len(parts) == 1:
        # Avoid over-triggering on sentence-initial common nouns like "Dog".
        return len(parts[0]) >= 4
    return False



def _dep_rel_base(rel: str) -> str:
    return (rel or "").split(":", 1)[0]


def _is_name_relation(rel: str) -> bool:
    return _dep_rel_base(rel) in _NAME_RELATIONS


def _name_component_indices(g: _DepGraph, head_idx: int) -> List[int]:
    """Return indices belonging to the same multiword name/compound component as `head_idx`.

    CoreNLP enhanced dependencies often subtype relations like 'flat:name'. This routine treats
    the base relation ('flat', 'compound', 'name') as the signal and traverses both parent and
    child edges to capture the full connected component.
    """
    seen: Set[int] = {head_idx}
    stack: List[int] = [head_idx]
    while stack:
        i = stack.pop()
        for rel, dep_idx in g.children(i):
            if _is_name_relation(rel) and dep_idx not in seen:
                seen.add(dep_idx)
                stack.append(dep_idx)
        for rel, gov_idx in g.parents(i):
            if _is_name_relation(rel) and gov_idx not in seen:
                seen.add(gov_idx)
                stack.append(gov_idx)
    return sorted(seen)


def _looks_like_proper_name(g: _DepGraph, head_idx: int) -> bool:
    tok = g.token(head_idx)
    if tok.pos.startswith("NNP"):
        return True
    if tok.ner not in ("O", ""):
        return True

    # Handle single-token hyphenated proper names like "Bill-Clinton".
    w = tok.word or ""
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        if len(parts) >= 2:
            cap_parts = sum(1 for p in parts if p[:1].isupper())
            if cap_parts >= 2:
                return True

    idxs = _name_component_indices(g, head_idx)
    if len(idxs) > 1:
        caps = 0
        for i in idxs:
            t = g.token(i)
            if t.pos.startswith("NNP") or t.ner not in ("O", "") or t.word[:1].isupper():
                caps += 1
        if caps >= 2:
            return True

    return False


def _extract_compound_text(g: _DepGraph, head_idx: int) -> str:
    idxs = _name_component_indices(g, head_idx)
    words = [g.token(i).word for i in idxs]
    return _normalize_name_text(" ".join(words))


def _det_for_head(g: _DepGraph, head_idx: int) -> Optional[str]:
    for rel, dep_idx in g.children(head_idx):
        if rel == "det" or rel.startswith(_DET_DEPS_PREFIX + ":"):
            return g.token(dep_idx).lemma.lower()
    return None


def _adjectives_for_head(g: _DepGraph, head_idx: int) -> List[str]:
    out: List[str] = []
    for rel, dep_idx in g.children_with_rel(head_idx, _AMOD_DEPS):
        out.append(g.token(dep_idx).lemma.lower())
    return out


def _relcl_heads_for_noun(g: _DepGraph, noun_head_idx: int) -> List[int]:
    out: List[int] = []
    for rel, dep_idx in g.children(noun_head_idx):
        if rel.startswith(_RELCL_DEPS_PREFIX) or rel in _RELCL_DEPS_FALLBACK:
            out.append(dep_idx)
    return out


def _find_subject(g: _DepGraph, head_idx: int) -> Optional[int]:
    for rel, dep_idx in g.children(head_idx):
        if rel in _SUBJECT_DEPS:
            return dep_idx
    return None


def _find_object(g: _DepGraph, head_idx: int) -> Optional[int]:
    for rel, dep_idx in g.children(head_idx):
        if rel in _OBJECT_DEPS or rel.startswith("obj"):
            return dep_idx
    return None


def _has_negation(g: _DepGraph, head_idx: int) -> bool:
    for rel, _ in g.children(head_idx):
        if rel in _NEG_DEPS:
            return True
    return False


# ----------------------------
# Translator
# ----------------------------

class CycLTranslator:
    """Compose CycL from CoreNLP annotation JSON."""

    def __init__(
        self,
        *,
        lexicon: Optional[CycLexicon] = None,
        query_mt: str = _DEFAULT_CYC_QUERY_MT,
        enable_scorer: bool = True,
        mapper: Optional[MappingProvider] = None,
    ):
        self._lexicon = lexicon or CycLexicon(None)
        self._query_mt = query_mt
        self._enable_scorer = bool(enable_scorer)

        self._mapper: MappingProvider = mapper or _default_mapping_provider(bridge=self._lexicon.bridge)
        self._lexicon.set_mapper(self._mapper)

        self._scorer: Optional[CycCandidateScorer] = None
        if self._enable_scorer and self._lexicon.bridge is not None:
            self._scorer = CycCandidateScorer(self._lexicon.bridge, query_mt=self._query_mt, mapper=self._mapper)

    @property
    def lexicon(self) -> CycLexicon:
        return self._lexicon

    def translate_annotation(self, ann: Dict[str, Any]) -> str:
        return cycl_to_string(self.translate_annotation_term(ann))

    def translate_annotation_term(self, ann: Dict[str, Any]) -> CycLTerm:
        sents = ann.get("sentences") or []
        if not sents:
            return "#$False"
        g = _DepGraph.from_corenlp_sentence(sents[0])
        alloc = _VarAllocator()

        body, bindings, _var_types = self._translate_sentence(g, alloc)

        formula = body
        for b in reversed(bindings):
            if b.quant == "exists":
                formula = _there_exists(b.var, _and(b.restrictor, formula))
            elif b.quant == "forall":
                formula = _for_all(b.var, _implies(b.restrictor, formula))
            elif b.quant == "no":
                formula = _for_all(b.var, _implies(b.restrictor, _not(formula)))
            else:
                formula = _there_exists(b.var, _and(b.restrictor, formula))

        return formula

    # -----------------------------------------------------------------
    # Assertion-mode translation (for KB augmentation)
    # -----------------------------------------------------------------

    def translate_to_assertions(
        self,
        ann: Dict[str, Any],
        *,
        entity_hint: Optional[str] = None,
    ) -> List[str]:
        """Translate a declarative CoreNLP annotation into ground CycL assertions.

        Unlike ``translate_annotation_term`` (which wraps variables in quantifiers
        to form a query), this method strips quantifiers and substitutes the known
        *entity_hint* constant for any unresolved subject variable, producing flat
        CycL sentences suitable for ``assert``.

        Parameters
        ----------
        ann : dict
            CoreNLP annotation JSON (must contain ``"sentences"``).
        entity_hint : str or None
            A ``#$``-prefixed Cyc constant representing the subject entity.
            When provided, any variable that occupies the *subject position* and
            that is not already a ground constant will be replaced with this value.

        Returns
        -------
        list[str]
            Zero or more fully-parenthesized CycL assertion strings with no free
            variables.  Each string can be passed directly to
            ``CycBridgeClient.assert_sentence``.
        """
        sents = ann.get("sentences") or []
        if not sents:
            return []

        all_assertions: List[str] = []
        for sent_data in sents:
            try:
                g = _DepGraph.from_corenlp_sentence(sent_data)
                alloc = _VarAllocator()
                body, bindings, var_types = self._translate_sentence(g, alloc)
            except Exception:
                continue

            grounded = self._ground_assertion(body, bindings, entity_hint=entity_hint)
            for term in grounded:
                s = cycl_to_string(term)
                # Skip trivial / tautological results.
                if s in ("#$True", "#$False"):
                    continue
                # Skip if any unresolved variables remain.
                if "?" in s:
                    continue
                all_assertions.append(s)

        return all_assertions

    def _ground_assertion(
        self,
        body: CycLTerm,
        bindings: List[_Binding],
        *,
        entity_hint: Optional[str] = None,
    ) -> List[CycLTerm]:
        """Convert (body, bindings) into a list of ground CycL terms.

        Strategy
        --------
        1. Build a substitution map  ``{var -> constant}``.
           * The first existentially-bound variable is assumed to be the subject;
             if *entity_hint* is supplied it replaces that variable.
        2. Apply the substitution to *body*.
        3. Extract additional ``(#$isa entity collection)`` assertions from any
           binding restrictor that mentions a meaningful collection type.
        """
        subst: Dict[str, str] = {}
        extra: List[CycLTerm] = []

        hint_used = False
        for b in bindings:
            var = b.var
            if entity_hint and not hint_used and b.quant == "exists":
                subst[var] = entity_hint
                hint_used = True
                # Pull isa-type facts out of the restrictor (e.g., (isa ?X Person))
                # and turn them into standalone assertions.
                for coll in self._extract_isa_collections(b.restrictor, var):
                    if coll not in ("#$True", "#$Thing", "#$Individual",
                                    "#$TemporalThing", "#$SomethingExisting"):
                        extra.append(["#$isa", entity_hint, coll])

        grounded_body = _subst_vars(body, subst)
        results: List[CycLTerm] = []

        body_s = cycl_to_string(grounded_body)
        if "?" not in body_s:
            results.append(grounded_body)

        for ea in extra:
            ea_s = cycl_to_string(ea)
            if "?" not in ea_s and ea_s != body_s:
                results.append(ea)

        return results

    @staticmethod
    def _extract_isa_collections(restrictor: CycLTerm, var: str) -> List[str]:
        """Pull collection constants from ``(#$isa var Collection)`` clauses inside a restrictor."""
        colls: List[str] = []
        _extract_isa_collections_rec(restrictor, var, colls)
        return colls

    def _maybe_translate_wh_is_proper_name(
        self, g: _DepGraph, alloc: _VarAllocator
    ) -> Optional[Tuple[CycLTerm, List[_Binding], Dict[str, str]]]:
        """Surface fallback for 'Who/What is <ProperName>?'.

        Some dependency variants (or tokenization quirks like hyphenated names) can cause the
        copular WH special-cases to miss the proper name, collapsing the query into a trivial
        existential. This routine uses only the token sequence as a guardrail fallback.
        """
        try:
            token_idxs = sorted(getattr(g, "_tokens", {}).keys())
        except Exception:
            token_idxs = []

        toks = [g.token(i) for i in token_idxs] if token_idxs else []
        # Filter out obvious punctuation tokens.
        nonpunct = [t for t in toks if (t.word or "") not in ("?", "!", ".", ",", ";", ":")]
        if len(nonpunct) < 3:
            return None

        first = nonpunct[0]
        second = nonpunct[1]

        wh_var, wh_kind = self._mapper.wh_info((first.lemma or "").lower())
        if not wh_var:
            wh_var, wh_kind = self._mapper.wh_info((first.word or "").lower())
        if not wh_var or wh_kind not in ("person", "thing", "which"):
            return None

        # Accept common copula surface forms.
        second_lemma = (second.lemma or "").lower()
        second_word = (second.word or "").lower()
        if second_lemma != "be" and second_word not in ("is", "are", "am", "was", "were", "be", "been", "being"):
            return None

        name_text = _normalize_name_text(" ".join((t.word or "") for t in nonpunct[2:]))
        if not name_text or not _text_seems_proper_name(name_text):
            return None

        ner = ""
        for t in nonpunct[2:]:
            if (t.ner or "") not in ("O", ""):
                ner = t.ner
                break

        subj_const = self._choose_proper_constant(text=name_text, ner=ner)

        ans_var = wh_var
        if ans_var == "?Who":
            ans_var = "?What"

        body: CycLTerm = ["#$isa", subj_const, ans_var]
        bindings: List[_Binding] = [_Binding(var=ans_var, quant="exists", restrictor=["#$isa", ans_var, "#$Collection"])]
        var_types: Dict[str, str] = {ans_var: "#$Collection"}
        return body, bindings, var_types

    def _translate_sentence(self, g: _DepGraph, alloc: _VarAllocator) -> Tuple[CycLTerm, List[_Binding], Dict[str, str]]:
        root = g.root()
        if root is None:
            return "#$False", [], {}

        wh_fallback = self._maybe_translate_wh_is_proper_name(g, alloc)
        if wh_fallback is not None:
            return wh_fallback

        # Copular: root is complement, with cop child and nsubj.
        if g.children_with_rel(root, _COPULA_DEPS):
            subj_idx = _find_subject(g, root)
            if subj_idx is None:
                return "#$False", [], {}

            subj_tok = g.token(subj_idx)
            comp_tok = g.token(root)

            # Special-case WH-copular questions about a proper name/entity:
            # "Who is Bill Clinton?" should ask for collections/types of Bill Clinton,
            # not for some ?Who that is a #$Clinton.
            subj_wh_var, subj_wh_kind = self._mapper.wh_info(subj_tok.lemma.lower())
            if not subj_wh_var:
                subj_wh_var, subj_wh_kind = self._mapper.wh_info(subj_tok.word.lower())

            if subj_wh_var and subj_wh_kind in ("person", "thing", "which") and _looks_like_proper_name(g, root) and not comp_tok.pos.startswith("JJ"):
                comp_text = _extract_compound_text(g, root)
                comp_const = self._choose_proper_constant(text=comp_text, ner=comp_tok.ner)

                ans_var = subj_wh_var
                if ans_var == "?Who":
                    ans_var = "?What"

                body = ["#$isa", comp_const, ans_var]
                bindings = [_Binding(var=ans_var, quant="exists", restrictor=["#$isa", ans_var, "#$Collection"])]
                var_types: Dict[str, str] = {ans_var: "#$Collection"}

                if _has_negation(g, root):
                    body = _not(body)
                return body, bindings, var_types

            # Also handle the common dependency arrangement for WH-copular questions
            # where the WH word is the copular complement/root and the proper name is the subject:
            # e.g., "Who is Bill Clinton?" can parse with root="who" and nsubj="Clinton".
            comp_wh_var, comp_wh_kind = self._mapper.wh_info(comp_tok.lemma.lower())
            if not comp_wh_var:
                comp_wh_var, comp_wh_kind = self._mapper.wh_info(comp_tok.word.lower())

            if comp_wh_var and comp_wh_kind in ("person", "thing", "which") and _looks_like_proper_name(g, subj_idx) and not subj_tok.pos.startswith("JJ"):
                subj_text = _extract_compound_text(g, subj_idx)
                subj_const = self._choose_proper_constant(text=subj_text, ner=subj_tok.ner)

                ans_var = comp_wh_var
                if ans_var == "?Who":
                    ans_var = "?What"

                body = ["#$isa", subj_const, ans_var]
                bindings = [_Binding(var=ans_var, quant="exists", restrictor=["#$isa", ans_var, "#$Collection"])]
                var_types: Dict[str, str] = {ans_var: "#$Collection"}

                if _has_negation(g, root):
                    body = _not(body)
                return body, bindings, var_types

            subj_term, subj_bindings, var_types = self._translate_np(g, subj_idx, alloc)

            if comp_tok.pos.startswith("JJ"):
                trait = self._choose_adj_trait(comp_tok.lemma, fallback_text=comp_tok.lemma)
                body = ["#$hasAttributeOrCollection", subj_term, trait]
            else:
                coll = self._choose_noun_collection(comp_tok.lemma, fallback_text=comp_tok.lemma)
                body = ["#$isa", subj_term, coll]

            if _has_negation(g, root):
                body = _not(body)
            return body, subj_bindings, var_types


        # Verb clause
        body, bindings, var_types = self._translate_verb_clause(g, root, alloc, subject_override=None)
        if _has_negation(g, root):
            body = _not(body)
        return body, bindings, var_types

    def _translate_verb_clause(
        self,
        g: _DepGraph,
        verb_idx: int,
        alloc: _VarAllocator,
        *,
        subject_override: Optional[str],
    ) -> Tuple[CycLTerm, List[_Binding], Dict[str, str]]:
        verb_tok = g.token(verb_idx)

        subj_idx = _find_subject(g, verb_idx)
        obj_idx = _find_object(g, verb_idx)

        bindings: List[_Binding] = []
        var_types: Dict[str, str] = {}

        if subject_override is not None:
            subj_term = subject_override
        elif subj_idx is not None:
            subj_term, b_s, t_s = self._translate_np(g, subj_idx, alloc)
            bindings.extend(b_s)
            var_types.update(t_s)
        else:
            subj_term = alloc.new_entity()
            bindings.append(_Binding(var=subj_term, quant="exists", restrictor="#$True"))

        obj_term: Optional[str] = None
        if obj_idx is not None:
            obj_term, b_o, t_o = self._translate_np(g, obj_idx, alloc)
            bindings.extend(b_o)
            var_types.update(t_o)

        transitive = obj_term is not None
        subj_type = var_types.get(subj_term) if subj_term.startswith("?") else None
        obj_type = var_types.get(obj_term) if (obj_term and obj_term.startswith("?")) else None

        formula = self._verb_formula(
            lemma=verb_tok.lemma,
            subj=subj_term,
            obj=obj_term,
            alloc=alloc,
            transitive=transitive,
            subj_type=subj_type,
            obj_type=obj_type,
        )
        return formula, bindings, var_types

    def _translate_np(self, g: _DepGraph, head_idx: int, alloc: _VarAllocator) -> Tuple[str, List[_Binding], Dict[str, str]]:
        tok = g.token(head_idx)
        lemma = tok.lemma.lower()

        pron = self._mapper.pronoun_var(lemma)
        if pron:
            return pron, [], {}

        wh_var, wh_kind = self._mapper.wh_info(lemma)
        if not wh_var:
            wh_var, wh_kind = self._mapper.wh_info(tok.word.lower())
        if wh_var:
            v = wh_var
            restrictor: CycLTerm = "#$True"
            var_types: Dict[str, str] = {}
            if wh_kind == "person":
                restrictor = ["#$isa", v, "#$Person"]
                var_types[v] = "#$Person"
            elif wh_kind == "place":
                restrictor = ["#$isa", v, "#$SpatialThing"]
                var_types[v] = "#$SpatialThing"
            elif wh_kind == "time":
                restrictor = ["#$isa", v, "#$TimeInterval"]
                var_types[v] = "#$TimeInterval"
            return v, [_Binding(var=v, quant="exists", restrictor=restrictor)], var_types

        det = _det_for_head(g, head_idx)
        det_type = self._mapper.determiner_type(det)
        quant = "exists"
        if det_type == "universal":
            quant = "forall"
        elif det_type == "negative":
            quant = "no"

        is_proper = _looks_like_proper_name(g, head_idx)
        if is_proper:
            text = _extract_compound_text(g, head_idx)
            const = self._choose_proper_constant(text=text, ner=tok.ner)
            return const, [], {}

        v = alloc.new_entity()
        coll = self._choose_noun_collection(lemma, fallback_text=lemma)
        restrictors: List[CycLTerm] = [["#$isa", v, coll]]
        var_types: Dict[str, str] = {v: coll}

        for adj in _adjectives_for_head(g, head_idx):
            trait = self._choose_adj_trait(adj, fallback_text=adj)
            restrictors.append(["#$hasAttributeOrCollection", v, trait])

        relcl_bindings: List[_Binding] = []
        for rel_verb_idx in _relcl_heads_for_noun(g, head_idx):
            rel_formula, rel_b, rel_types = self._translate_verb_clause(g, rel_verb_idx, alloc, subject_override=v)
            restrictors.append(rel_formula)
            relcl_bindings.extend(rel_b)
            var_types.update(rel_types)

        binding = _Binding(var=v, quant=quant, restrictor=_and(*restrictors))
        return v, [binding, *relcl_bindings], var_types

    def _choose_noun_collection(self, lemma: str, *, fallback_text: str) -> str:
        lemma_l = (lemma or "").lower()
        cands = self._lexicon.noun_candidates(lemma_l)
        if not cands:
            fb = self._mapper.fallback_noun_collections(lemma_l)
            if fb:
                return fb[0]
            return _as_cyc_constant_from_text(fallback_text)
        if self._scorer is None:
            return cands[0]
        scored = [(self._scorer.score_noun_collection(c, lemma=lemma_l), c) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _choose_adj_trait(self, lemma: str, *, fallback_text: str) -> str:
        lemma_l = (lemma or "").lower()
        traits, _templates = self._lexicon.adj_candidates(lemma_l)
        if not traits:
            fb = self._mapper.fallback_adj_traits(lemma_l)
            if fb:
                return fb[0]
            return _as_cyc_constant_from_text(fallback_text)
        return traits[0]

    def _choose_proper_constant(self, *, text: str, ner: str) -> str:
        raw_t = (text or "").strip()
        t = _normalize_name_text(raw_t)
        if not t:
            t = raw_t
        if not t:
            return _as_cyc_constant_from_text("Thing")

        # Prefer Cyc nameString/preferredNameString lookups when possible.
        cands = self._lexicon.proper_candidates(t) if self._lexicon.bridge is not None else []
        if not cands:
            return _as_cyc_constant_from_text(t)
        if self._scorer is None:
            return cands[0]
        scored = [(self._scorer.score_proper_term(c, text=t, ner=ner), c) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _verb_formula(
        self,
        *,
        lemma: str,
        subj: str,
        obj: Optional[str],
        alloc: _VarAllocator,
        transitive: bool,
        subj_type: Optional[str],
        obj_type: Optional[str],
    ) -> CycLTerm:
        lemma_l = (lemma or "").lower()
        denots, templates_raw = self._lexicon.verb_candidates(lemma_l)

        templates: List[CycLTerm] = []
        for tr in templates_raw:
            try:
                templates.append(parse_cycl_sexp(tr))
            except Exception:
                continue

        if templates:
            ranked: List[Tuple[float, CycLTerm]] = []
            for t in templates:
                score = 0.0
                txt = cycl_to_string(t)
                if ":SUBJECT" in txt:
                    score += 2.0
                if transitive and ":OBJECT" in txt:
                    score += 2.0
                if transitive and ":OBJECT" not in txt:
                    score -= 3.0
                ranked.append((score, t))
            ranked.sort(key=lambda x: x[0], reverse=True)
            for _, t in ranked:
                inst = self._instantiate_verb_template(t, subj=subj, obj=obj, alloc=alloc)
                if inst is not None:
                    return inst

        pred = None
        if denots:
            if self._scorer is None:
                pred = denots[0]
            else:
                scored = [
                    (
                        self._scorer.score_verb_predicate(
                            d,
                            lemma=lemma_l,
                            transitive=transitive,
                            subj_type=subj_type,
                            obj_type=obj_type,
                        ),
                        d,
                    )
                    for d in denots
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                pred = scored[0][1]
        if not pred:
            fb = self._mapper.fallback_verb_predicates(lemma_l)
            pred = fb[0] if fb else _as_cyc_constant_from_text(lemma_l)

        if transitive and obj is not None:
            return [pred, subj, obj]
        return [pred, subj]

    def _instantiate_verb_template(self, template: CycLTerm, *, subj: str, obj: Optional[str], alloc: _VarAllocator) -> Optional[CycLTerm]:
        mapping: Dict[str, str] = {":SUBJECT": subj}
        if obj is not None:
            mapping[":OBJECT"] = obj

        txt = cycl_to_string(template)
        needs_event = False
        event_var = ""
        if ":ACTION" in txt:
            needs_event = True
            event_var = alloc.new_event()
            mapping[":ACTION"] = event_var
        elif ":EVENT" in txt:
            needs_event = True
            event_var = alloc.new_event()
            mapping[":EVENT"] = event_var

        inst, _used = _substitute_placeholders(template, mapping)
        if _contains_unresolved_placeholder(inst):
            return None
        if needs_event and event_var:
            return _there_exists(event_var, inst)
        return inst


# ----------------------------
# CLI (standalone)
# ----------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="CoreNLP -> CycL translator (rule-based)")
    ap.add_argument("--text", required=True, help="English text to translate")
    ap.add_argument("--corenlp", default=_DEFAULT_CORENLP_URL, help="CoreNLP server base URL")
    ap.add_argument("--cyc-bridge", default=_DEFAULT_CYC_BRIDGE_URL, help="Cyc bridge base URL")
    ap.add_argument("--lex-mt", default=_DEFAULT_CYC_LEXICON_MT, help="Cyc lexicon microtheory")
    ap.add_argument("--query-mt", default=_DEFAULT_CYC_QUERY_MT, help="Cyc query microtheory for scoring checks")
    ap.add_argument("--no-cyc-lex", action="store_true", help="Disable Cyc lexicon lookups")
    ap.add_argument("--no-cyc-score", action="store_true", help="Disable Cyc scoring/ranking")
    ap.add_argument("--use-ollama-mappings", action="store_true", help="Use Ollama for controlled token/lexicon mappings (overrides env)")
    ap.add_argument("--no-ollama-mappings", action="store_true", help="Disable Ollama mappings (overrides env)")
    ap.add_argument("--ollama", default=_DEFAULT_OLLAMA_BASE_URL, help="Ollama base URL (default from env OLLAMA_BASE_URL)")
    ap.add_argument("--ollama-model", default=_DEFAULT_OLLAMA_MODEL, help="Ollama model name (default from env OLLAMA_MODEL)")
    ap.add_argument("--ollama-temp", type=float, default=_DEFAULT_OLLAMA_TEMPERATURE, help="Ollama temperature (default from env OLLAMA_TEMPERATURE)")
    ap.add_argument("--ollama-timeout", type=int, default=_DEFAULT_OLLAMA_TIMEOUT_SEC, help="Ollama timeout seconds (default from env OLLAMA_TIMEOUT_SEC)")
    args = ap.parse_args()

    nlp = CoreNLPServerClient(args.corenlp)
    bridge = None if args.no_cyc_lex else CycBridgeClient(args.cyc_bridge)

    mapper: Optional[MappingProvider] = None
    if args.use_ollama_mappings and not args.no_ollama_mappings:
        mapper = OllamaMappingProvider(
            base_url=args.ollama,
            model=args.ollama_model,
            temperature=float(args.ollama_temp),
            timeout_sec=int(args.ollama_timeout),
            bridge=bridge,
        )
    elif args.no_ollama_mappings:
        mapper = StaticMappingProvider()

    lex = CycLexicon(bridge=bridge, lex_mt=args.lex_mt, lex_limit=_DEFAULT_CYC_LEX_LIMIT, mapper=mapper)
    tr = CycLTranslator(lexicon=lex, query_mt=args.query_mt, enable_scorer=not args.no_cyc_score, mapper=mapper)

    ann = nlp.annotate(args.text)
    print(tr.translate_annotation(ann))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
