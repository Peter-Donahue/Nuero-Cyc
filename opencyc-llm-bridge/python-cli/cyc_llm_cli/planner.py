from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


def plan_json_schema() -> Dict[str, Any]:
    """
    JSON schema enforced by Ollama structured outputs.
    Keep it simple (no oneOf) to maximize model compliance.
    """
    return {
        "type": "object",
        "properties": {
            "analysis_summary": {"type": "string"},
            "session_mt": {"type": "string"},
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["ensure_term", "assert", "ask_true", "ask_var", "converse"],
                        },
                        # ensure_term
                        "name": {"type": "string"},
                        "surface": {"type": "string"},
                        # assert
                        "sentence": {"type": "string"},
                        # ask_true / ask_var
                        "query": {"type": "string"},
                        "var": {"type": "string"},
                        "limit": {"type": "integer"},
                        # all actions
                        "mt": {"type": "string"},
                        # converse
                        "subl": {"type": "string"},
                    },
                    "required": ["type"],
                    "additionalProperties": False,
                },
            },
            "final_response_instructions": {"type": "string"},
        },
        "required": ["session_mt", "actions", "final_response_instructions"],
        "additionalProperties": False,
    }


def query_json_schema() -> Dict[str, Any]:
    """Schema for the initial translation step: emit only a single OpenCyc query."""
    return {
        "type": "object",
        "properties": {
            "analysis_summary": {"type": "string"},
            "query_type": {"type": "string", "enum": ["ask_var", "ask_true"]},
            "query": {"type": "string"},
            "var": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query_type", "query", "var", "limit"],
        "additionalProperties": False,
    }


def answer_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "cyc_evidence": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "cyc_evidence", "limitations"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class PlannerContext:
    session_mt: str
    session_genl_mt: str
    debug: bool = False


def build_query_messages(
    *,
    ctx: PlannerContext,
    user_prompt: str,
    conversation: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Build messages for the *first attempt* CycL translation call.

    This intentionally outputs only a single query spec, and does not allow the model
    to pre-assert knowledge into the session MT.
    """

    schema_str = json.dumps(query_json_schema(), indent=2)

    system = f"""You are a CycL translator. You do NOT answer the user's question.

Task:
- Convert the user's prompt into a SINGLE CycL query suitable for OpenCyc.
- Output MUST be ONLY JSON that validates against the schema below.

Important:
- This is the FIRST attempt. Do NOT propose ensure_term or assert steps here.
  Only produce the query itself.

Session microtheory (mt) is handled by the tool runner separately: {ctx.session_mt}
Do NOT wrap queries with #$ist.

Query selection:
- Use ask_true ONLY for yes/no questions ("is/are/was/were/do/does/can...?").
- For "what/which/how many/when/who" questions, use ask_var with exactly ONE variable.

CycL rules:
- Query MUST be one fully-parenthesized CycL sentence, e.g. (#$isa #$Dog #$Animal)
- Predicates and constants MUST be #$-prefixed Cyc constants.
- Variables are ?X, ?N, ?THING, etc and appear inside the query (no outer binders like (?x) ...).

Entity naming (when you don't know the official Cyc constant name):
- Use a concrete CamelCase #$Constant derived from the prompt (e.g., #$StarTrekOriginalSeries),
  NOT placeholders like #$SomeSeries / #$SomeThing.

Predicate naming:
- Prefer existing Cyc predicates when plausible (#$isa, #$genls, #$and, #$or, #$not, etc).
- If you must invent a domain predicate, make it CamelCase (e.g., #$EpisodeCountOfSeries).

Output schema:
{schema_str}
"""

    convo_tail = conversation[-8:] if conversation else []
    convo_text = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in convo_tail])

    user = f"""USER PROMPT:
{user_prompt}

CONVERSATION CONTEXT (most recent last):
{convo_text}
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_planner_messages(
    *,
    ctx: PlannerContext,
    user_prompt: str,
    conversation: List[Dict[str, str]],
    allow_converse: bool,
) -> List[Dict[str, str]]:
    schema_str = json.dumps(plan_json_schema(), indent=2)

    system = f"""You are a planner that produces a strict JSON plan for interacting with OpenCyc.
You do NOT answer the user's question directly.

You have access to these actions (via an external tool runner):
- ensure_term: check if a Cyc constant exists; create it if missing.
- assert: add a CycL sentence into the session microtheory.
- ask_true: ask if a CycL query is true in the session microtheory.
- ask_var: ask a CycL query with ONE variable and return bindings for that variable.
- converse: run raw SubL (use only if absolutely needed; may be disabled).

Core goal:
- Produce a plan whose *final* ask_true/ask_var result directly contains the answer, so the answer stage can
  simply report what OpenCyc returned (no guessing in the final stage).

Rules:
- The session microtheory is: {ctx.session_mt}
- You MUST put mt=\"{ctx.session_mt}\" on every action you generate (except converse which can omit mt).
- Prefer querying OpenCyc over using your own knowledge.
- If the KB doesn't contain a needed fact, you MAY assert a *minimal* supplemental fact in the session MT,
  then re-query so OpenCyc returns the answer from the KB/session MT.
- If you create new constants with ensure_term, those constants MUST be used later in an assert/query
  (do not create unused placeholders).
- Do NOT create constants whose names start with \"ensure_\". \"ensure_term\" is the action name; the constant
  name itself should be the actual Cyc constant you intend to use (prefer CamelCase).
- Keep assertions minimal; do NOT dump large ontologies.
- Output MUST validate against this JSON schema (and must be ONLY JSON, no prose):

CycL formatting rules (must follow):
- All CycL sentences/queries MUST be fully parenthesized, e.g.: (#$isa #$Dog #$Animal)
- Predicates must be Cyc constants starting with #$ (e.g., #$genls, #$isa). Built-in predicates are often lowercase, but newly created predicates can be CamelCase.
- Constants must start with #$ (e.g., #$Dog, #$Animal). No :DOG, no \"DOG\", no (QUOTE ...).
- Numbers can be literal integers (e.g., 79).
- Do NOT wrap queries/sentences with #$ist. The tool already takes mt separately.
- Do NOT use outer variable binders like (?x) ...; variables belong *inside* the CycL sentence as ?X, ?N, etc.
- Only use #$isa when the subject is an individual (a specific named entity), not a type.
- Prefer ask_var / ask_true; end with an ask_var/ask_true that directly answers the user's question.
  - Use ask_true ONLY for yes/no questions. For \"how many / what / which\" questions, end with ask_var.
  
CycL (EL) Reference — for generating OpenCyc queries/assertions

SCOPE
- Use EL CycL (Epistemological Level): first-order predicate calculus style expressions, written as Lisp-like S-expressions.
- Avoid HL-oriented operators (e.g., #$LogAnd / #$LogOr) unless you know they are required.
- Microtheory context (Mt) is supplied out-of-band by the tool. DO NOT wrap queries or assertions in (#$ist <mt> ...).

LEXICAL / TOKENS
- Parentheses denote application: (RELATION ARG1 ARG2 ...).
- Constants: begin with "#$" (e.g., #$Dog, #$UnitedStates, #$WorldLeader).
  - Constant names are typically CamelCase and may include hyphens/digits, but must be a single symbol token.
- Variables: begin with "?" (e.g., ?X, ?COUNTRY, ?PERSON). Do NOT use $x / $y.
- Numbers: use bare numerals (e.g., 25, 3, 79).
- Strings: use double quotes for SubLString arguments, escaping internal quotes/backslashes as needed.
  Example: (#$comment #$SomeTerm "Text with \\\"quotes\\\" and \\\\ backslashes")

TERMS (what can appear as arguments)
1) Constant term: #$Mexico
2) Variable term: ?X
3) Literal term: 2000, 79, "some text"
4) Functional term (function application yields a TERM):
   - Syntax: (<FunctionConst> <arg1> ... <argN>)
   - Examples:
     (#$PresidentFn #$Mexico)
     (#$MotherFn (#$PresidentFn #$UnitedStates))
     (#$YearFn 2000)

ATOMIC SENTENCES (GAF-like)
- Syntax: (<PredicateConst> <arg1> ... <argN>)
- Examples:
  (#$isa #$GeorgeWBush #$WorldLeader)
  (#$genls #$Cat #$Carnivore)

COLLECTIONS VS INDIVIDUALS (MOST IMPORTANT MODELING RULE)
- #$genls relates two COLLECTIONS: “Subcollection / specialization”
  - (#$genls #$Dog #$CanineAnimal)   ; dogs are a kind of canine animal
- #$isa relates a THING to a COLLECTION: “membership / instance-of”
  - (#$isa #$GeorgeWBush #$WorldLeader) ; an individual in a collection
  - NOTE: The THING can itself be a collection when asserting that a collection is an instance of a higher-level collection
          (i.e., a collection-of-collections classification).
- When in doubt:
  - Use #$genls for “X is a kind/type of Y” (taxonomy between collections).
  - Use #$isa for “this entity is a member of that collection” (membership).

LOGICAL CONNECTIVES (build compound sentences)
- Conjunction:
  (#$and S1 S2 ... Sn)
- Disjunction:
  (#$or S1 S2 ... Sn)
- Negation:
  (#$not S)
- Implication (rule-like):
  (#$implies ANTECEDENT CONSEQUENT)

QUANTIFIERS (bind variables)
- Universal quantification:
  (#$forAll ?VAR SENTENCE)
  For multiple vars, nest quantifiers:
  (#$forAll ?X (#$forAll ?Y ...))
- Existential quantification:
  (#$thereExists ?VAR SENTENCE)
  Also nest as needed.

EQUALITY / COMPARISON (use only if needed)
- Equality:
  (#$equals T1 T2)
- Numeric comparisons (example):
  (#$greaterThan 25 3)

RELATION / WELL-FORMEDNESS META-VOCAB (use sparingly)
- Relations have constraints; if you must define/repair a relation term, these exist:
  - (#$arity <Pred> <N>)
  - (#$arg1Isa <Pred> <Collection>), (#$arg2Isa ...), etc.
- Prefer using existing predicates over creating new ones.

QUERYING GUIDELINES FOR THIS APP
- ask_true:
  - Supply a CLOSED sentence (no free variables), or ensure any variables are bound by #$forAll/#$thereExists.
- ask_var:
  - Supply a query with EXACTLY ONE free variable (e.g., ?X).
  - Set action.var to that exact variable string (e.g., "?X").
  - Example:
    query = "(#$isa ?X #$Dog)"
    var   = "?X"
- Do NOT use #$ist inside query strings; mt is supplied by the tool runner separately.

ASSERTION GUIDELINES (if/when you assert)
- Assert minimal, safe EL sentences only.
- Prefer typing links:
  - If asserting taxonomy: (#$genls #$NewType #$SomeParentType)
  - If asserting membership: (#$isa #$NewThing #$SomeCollection)
- Avoid inventing large ontologies or many new predicates.

If you need to represent a missing scalar value and you cannot find an existing Cyc predicate:
- Create/ensure BOTH:
  - a predicate constant (CamelCase) for the relationship, e.g., EpisodeCountOfSeries
  - the concrete entity constant(s) from the user's question (e.g., StarTrekOriginalSeries)
- IMPORTANT: Do NOT use generic placeholders like #$SomeSeries, #$SomeThing, #$SomeEntity.
- Use separate assert actions for typing/value facts (do NOT put typing facts in ensure_term).
- Then assert the predicate type, assert the value, and re-query so OpenCyc returns the value from the session MT.

{schema_str}
"""

    # Provide short conversation context, but keep it small.
    convo_tail = conversation[-8:] if conversation else []
    convo_text = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in convo_tail])

    user = f"""USER PROMPT:
{user_prompt}

CONVERSATION CONTEXT (most recent last):
{convo_text}

NOTE:
- If you want to use converse but allow_converse={str(allow_converse).lower()}, you must NOT emit any converse actions when allow_converse=false.
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_answer_messages(
    *,
    user_prompt: str,
    cyc_results: Dict[str, Any],
    authoritative_answer: str,
    conversation: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    schema_str = json.dumps(answer_json_schema(), indent=2)

    system = f"""You translate OpenCyc results into a clear, accurate natural language answer.

Hard rules:
- Base your answer ONLY on the provided Cyc results and logical implications of those results.
- The field `answer` MUST be exactly the provided authoritative answer string (verbatim) so the UI can display
  the tool-derived value without any model substitution.
- If the Cyc results are insufficient, say so plainly in limitations.
- Do not invent Cyc facts.
- Keep the answer readable and concise.

Output MUST be ONLY JSON and must validate against this schema:

{schema_str}
"""

    convo_tail = conversation[-8:] if conversation else []
    convo_text = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in convo_tail])

    user = f"""USER PROMPT:
{user_prompt}

CONVERSATION CONTEXT (most recent last):
{convo_text}

CYC RESULTS (authoritative):
{json.dumps(cyc_results, indent=2)}

AUTHORITATIVE_ANSWER (verbatim; `answer` must match exactly):
{authoritative_answer}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
