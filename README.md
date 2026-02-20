# Nuero-Cyc

I came across an obituary for Douglas Lenat (creator of Cyc) and fell down the rabbit hole into Cyc / OpenCyc.

- Economist obituary: https://www.economist.com/obituary/2023/09/13/douglas-lenat-trained-computers-to-think-the-old-fashioned-way
- OpenCyc fork / last version: https://github.com/Peter-Donahue/opencyc
- CoreNLP fork: https://github.com/Peter-Donahue/CoreNLP
  
This repo is still a prototype: limited natural-language coverage, rough edges, and plenty of “works for the cases I tried”.

## What this project does now

This project provides a CLI that lets you type **plain English** and run it against **OpenCyc**.

Key change vs earlier versions:  
**English → CycL is no longer done by a general-purpose LLM planner.** Instead it is done by:

- **Stanford CoreNLP** for tokenization/POS/NER/dependencies
- A **deterministic rule-based translator** (`corenlp_to_cycl.py`) that composes a CycL query
- Optional **Cyc-backed lexicon mapping + scoring** to map English lemmas to Cyc constants/predicates
- Optional (controlled) **Ollama** calls only for small mapping decisions (NOT answer writing)

Also:  
**There is no LLM post-processing of OpenCyc’s answers anymore.** Output is rendered using Cyc’s own lexical facts (e.g., `preferredNameString` / `nameString`) and otherwise falls back to raw Cyc terms.

## High-level architecture

### Components
- **OpenCyc server** (reasoning + KB)
- **Java bridge server** (`CycBridgeServer`)  
  Exposes HTTP endpoints like `ask_true`, `ask_var`, `assert`, plus optional lexicon lookup utilities.
- **Python CLI** (`cyc_llm_cli/`)  
  - calls CoreNLP for parsing
  - calls `corenlp_to_cycl.py` to build CycL
  - queries OpenCyc via the Java bridge
  - renders results using Cyc’s own name strings

### Query flow
1. You type an English prompt in the CLI.
2. CoreNLP produces tokens/lemmas/POS/NER + dependency parse.
3. `corenlp_to_cycl.py` composes a CycL formula (quantifiers, `#$isa`, basic predicate application, some negation/coordination/relative clauses).
4. Optional: the translator queries Cyc’s lexicon (denotations / templates) and scores candidates with cheap `isa/genls/arity/argIsa/nameString` checks.
5. The orchestrator decides:
   - **`ask_var`** if there’s a WH query variable (`?Who/?What/?Which/?Where/?When`)
   - otherwise **`ask_true`**
6. Queries run in a **session microtheory** so any session-scoped constant creation/assertions don’t pollute the KB.
7. Results are printed using Cyc’s own lexical strings where available.

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

### 4) Python CLI
Create a venv and install dependencies, then run the CLI module.

## Running

### Start services
1) Start OpenCyc  
2) Start the Java bridge server (HTTP on `:8081`)  
3) Start CoreNLP server (HTTP on `:9000`)

### Run the CLI
From the Python CLI folder, run whatever launcher script you have (e.g., `run.ps1`) or:

```bash
python -m cyc_llm_cli
