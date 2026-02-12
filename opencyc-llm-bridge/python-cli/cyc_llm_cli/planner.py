from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
                            "enum": ["ensure_term", "assert", "ask_true", "ask_var", "converse"]
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
                    "additionalProperties": False
                }
            },
            "final_response_instructions": {"type": "string"},
        },
        "required": ["session_mt", "actions", "final_response_instructions"],
        "additionalProperties": False
    }


def answer_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "cyc_evidence": {
                "type": "array",
                "items": {"type": "string"}
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"}
            },
        },
        "required": ["answer", "cyc_evidence", "limitations"],
        "additionalProperties": False
    }


@dataclass(frozen=True)
class PlannerContext:
    session_mt: str
    session_genl_mt: str
    debug: bool = False


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
- You MUST put mt="{ctx.session_mt}" on every action you generate (except converse which can omit mt).
- Prefer querying OpenCyc over using your own knowledge.
- If the KB doesn't contain a needed fact, you MAY assert a *minimal* supplemental fact in the session MT,
  then re-query so OpenCyc returns the answer from the KB/session MT.
- If you create new constants with ensure_term, those constants MUST be used later in an assert/query
  (do not create unused placeholders).
- Do NOT create constants whose names start with "ensure_". "ensure_term" is the action name; the constant
  name itself should be the actual Cyc constant you intend to use (prefer CamelCase).
- Keep assertions minimal; do NOT dump large ontologies.
- Output MUST validate against this JSON schema (and must be ONLY JSON, no prose):

CycL formatting rules (must follow):
- All CycL sentences/queries MUST be fully parenthesized, e.g.: (#$isa #$Dog #$Animal)
- Predicates must be Cyc constants like #$genls and #$isa (lowercase, with #$).
- Constants must start with #$ (e.g., #$Dog, #$Animal). No :DOG, no "DOG", no (QUOTE ...).
- Numbers can be literal integers (e.g., 79).
- Do NOT wrap queries/sentences with #$ist. The tool already takes mt separately.
- Do NOT use outer variable binders like (?x) ...; variables belong *inside* the CycL sentence as ?X, ?N, etc.
- Only use #$isa when the subject is an individual (a specific named entity), not a type.
- Prefer ask_var / ask_true; end with an ask_var/ask_true that directly answers the user's question.
  - Use ask_true ONLY for yes/no questions. For "how many / what / which" questions, end with ask_var.

If you need to represent a missing scalar value and you cannot find an existing Cyc predicate:
- Create a predicate constant (ensure_term) like EpisodeCountOfSeries (CamelCase),
  assert its type as a binary predicate, then assert the value and re-query:
  1) ensure_term: name=EpisodeCountOfSeries
  2) assert: sentence=(#$isa #$EpisodeCountOfSeries #$BinaryPredicate)
  3) assert: sentence=(#$EpisodeCountOfSeries #$SomeSeries 79)
  4) ask_var: query=(#$EpisodeCountOfSeries #$SomeSeries ?N) var=?N
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
