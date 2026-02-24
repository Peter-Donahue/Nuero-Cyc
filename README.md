# Neuro-Cyc

I came across an obituary for Douglas Lenat (creator of Cyc) and fell down the rabbit hole into Cyc / OpenCyc.

- Economist obituary: https://www.economist.com/obituary/2023/09/13/douglas-lenat-trained-computers-to-think-the-old-fashioned-way
- OpenCyc fork / last version: https://github.com/Peter-Donahue/opencyc
- CoreNLP fork: https://github.com/Peter-Donahue/CoreNLP
  
This repo is still a prototype: limited natural-language coverage, rough edges, and plenty of "works for the cases I tried".

## What this project does now

This project provides a CLI that lets you type **plain English** and run it against **OpenCyc**.

Key design decisions:

**English → CycL is not done by a general-purpose LLM planner.** Instead it is done by:

- **Stanford CoreNLP** for tokenization/POS/NER/dependencies
- A **deterministic rule-based translator** (`corenlp_to_cycl.py`) that composes a CycL query from the dependency parse
- Optional **Cyc-backed lexicon mapping + scoring** to map English lemmas to Cyc constants/predicates
- Optional (controlled) **Ollama** calls only for small mapping decisions (NOT answer generation)

**There is no LLM post-processing of OpenCyc's answers.** Output is rendered using Cyc's own lexical facts (e.g., `preferredNameString` / `nameString`) and otherwise falls back to raw Cyc terms.

**When OpenCyc lacks knowledge, an LLM fills the gap — but through the NLP pipeline, not by writing CycL directly.** If a query returns no results, the system asks Ollama for simple English facts, then runs those facts back through CoreNLP → `corenlp_to_cycl.py` to produce well-formed CycL assertions. These are asserted into a session microtheory, and the original query is re-run. The LLM never sees or writes the final CycL — it only supplies declarative English sentences and a small number of direct predicate assertions (for things the NLP pipeline can't express, like free-text comments and dates).

## High-level architecture

### Components
- **OpenCyc server** (reasoning + KB)
- **Java bridge server** (`CycBridgeServer`)  
  Exposes HTTP endpoints like `ask_true`, `ask_var`, `assert`, plus optional lexicon lookup utilities.
- **Stanford CoreNLP server** (tokenization, POS, NER, dependency parsing)
- **Ollama** (local LLM, optional) — used for controlled lexicon mapping during translation, and for KB augmentation when queries return empty results
- **Python CLI** (`cyc_llm_cli/`)  
  - `corenlp_to_cycl.py` — rule-based translator (CoreNLP annotation → CycL)
  - `orchestrator.py` — query routing, KB augmentation, result rendering
  - `cyc_bridge_client.py` — HTTP client for the Java bridge
  - `cli.py` — Rich-based terminal UI
  - `config.py` — settings with env-var and CLI-flag overrides

### Query flow

```
  ┌─────────────────┐
  │  English prompt  │
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │    CoreNLP       │  tokenize, POS, NER, depparse
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ corenlp_to_cycl  │  dependency-driven CycL composition
  │  (rule-based)    │  + Cyc lexicon lookup + scoring
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  Orchestrator    │  query routing: ask_var or ask_true
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │    OpenCyc       │  query against KB
  └────────┬────────┘
           │
           ├── results found ──► render via Cyc lexical strings ──► display
           │
           └── no results ──► LLM KB augmentation (see below) ──► re-query
```

1. You type an English prompt in the CLI.
2. CoreNLP produces tokens, lemmas, POS tags, NER labels, and a dependency parse.
3. `corenlp_to_cycl.py` composes a CycL formula from the dependency tree — quantifiers, `#$isa`, predicate application, negation, coordination, and relative clauses.
4. The translator optionally queries Cyc's lexicon (denotations / templates) and scores candidates with cheap `isa`/`genls`/`arity`/`argIsa`/`nameString` checks to pick the best constant for each English token.
5. The orchestrator decides the query type:
   - **`ask_var`** if there's a WH query variable (`?Who` / `?What` / `?Which` / `?Where` / `?When`)
   - **`ask_true`** otherwise
6. For "Who is X?" questions, the orchestrator runs a **multi-strategy entity description**: querying `#$comment`, known predicates (`#$occupation`, `#$spouse`, `#$birthDate`, etc.), and filtered `#$isa` (stripping abstract internal types like `Thing`, `Agent-Generic`, `TemporallyExistingThing`).
7. All queries run in a **session microtheory** so constant creation and assertions don't pollute the base KB.
8. Results are rendered using Cyc's own lexical strings where available.

### LLM KB augmentation (dual-path)

When a query returns no meaningful results and Ollama is available, the orchestrator asks the LLM for factual information about the entity or topic. The LLM returns two things:

**Path 1 — English sentences through the NLP pipeline:**
```
LLM produces:  "Bill Clinton is a president."
                         │
                         ▼
                CoreNLP annotate
                         │
                         ▼
          corenlp_to_cycl.translate_to_assertions(entity_hint=#$BillClinton)
                         │
                         ▼
          (#$isa #$BillClinton #$President)   ◄── grounded assertion
          (#$isa #$BillClinton #$Person)      ◄── bonus from restrictor
                         │
                         ▼
                assert into session MT
```

The translator's `translate_to_assertions()` method works like the normal query translation, but instead of wrapping variables in quantifiers, it **grounds** them by substituting the known entity constant. This reuses the full Cyc lexicon lookup and scoring — so "president" resolves to whatever `#$President` or `#$UnitedStatesPresident` constant the lexicon finds, not whatever the LLM hallucinated.

**Path 2 — Direct CycL for predicates the pipeline can't express:**
```
LLM produces:  (#$comment #$BillClinton "42nd President of the United States")
                         │
                         ▼
                normalize (add missing #$ prefixes)
                         │
                         ▼
                ensure constants exist (create if needed)
                         │
                         ▼
                assert into session MT
```

This path handles string-valued predicates (`#$comment`), dates (`#$birthDate`), and other facts that don't map to simple "X is a Y" English sentences. A normalizer fixes common LLM issues like missing `#$` prefixes on constants.

After both paths complete, the **original query is re-run** against the now-augmented session MT. The debug panel shows every step: which sentences the LLM produced, what CycL the pipeline generated from each, which assertions succeeded or failed, and the re-query results.

## Requirements

### 1) OpenCyc
You need a running OpenCyc image. See the fork above.

Typical defaults used by the bridge:
- Cyc host: `localhost`
- Cyc port: `3601`

### 2) Java bridge server
Run the Java bridge that exposes HTTP endpoints used by the Python CLI (defaults to `http://localhost:8081`).

### 3) Stanford CoreNLP server
You need CoreNLP running in server mode (defaults to `http://localhost:9000`).

Recommended annotators:
- `tokenize,ssplit,pos,lemma,ner,depparse`

### 4) Ollama (optional, recommended)
For LLM KB augmentation and controlled lexicon mapping. Defaults to `http://localhost:11434`.

```bash
ollama pull llama3
ollama serve
```

The system gracefully degrades if Ollama is unavailable — queries still work, they just can't augment missing KB data.

### 5) Python CLI
Create a venv and install dependencies, then run the CLI module.

## Running

### Start services
1) Start OpenCyc  
2) Start the Java bridge server (HTTP on `:8081`)  
3) Start CoreNLP server (HTTP on `:9000`)  
4) Start Ollama (optional, HTTP on `:11434`)

### Run the CLI

```bash
python -m cyc_llm_cli
```

Single question:
```bash
python -m cyc_llm_cli --once "Who is Bill Clinton?"
```

With debug output:
```bash
python -m cyc_llm_cli --debug --once "Who is Bill Clinton?"
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--debug` | Show full debug panel (query translation, execution log, LLM augmentation details) |
| `--once "..."` | Run one question and exit |
| `--llm-aug` | Enable LLM KB augmentation (on by default) |
| `--no-llm-aug` | Disable LLM KB augmentation |
| `--ollama URL` | Ollama base URL (default: `http://localhost:11434`) |
| `--ollama-model NAME` | Ollama model (default: `llama3`) |
| `--bridge URL` | Cyc bridge base URL (default: `http://localhost:8081`) |
| `--corenlp URL` | CoreNLP base URL (default: `http://localhost:9000`) |
| `--no-cyc-lex` | Disable Cyc lexicon lookups |
| `--no-cyc-score` | Disable Cyc candidate scoring |
| `--no-cyc-nl` | Disable Cyc term→English rendering for results |

### Environment variables

All settings can also be controlled via environment variables: `CYC_BRIDGE_BASE_URL`, `CORENLP_BASE_URL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `USE_LLM_KB_AUGMENTATION`, etc. See `config.py` for the full list.
