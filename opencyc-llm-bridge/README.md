# OpenCyc ⇄ Local LLM Bridge (CLI-first)

This repository is a basic, end-to-end implementation of:

1) A tiny local HTTP service that wraps the legacy **OpenCyc Java API** (so you can talk to OpenCyc over HTTP/JSON), and  
2) A **Python CLI** that calls a **local LLM** (Ollama by default) to:
   - translate user prompts into a Cyc “plan”
   - check/extend a *session microtheory*
   - ask OpenCyc to perform reasoning
   - translate OpenCyc results back into readable answers

This is intentionally a *starter implementation*:
- “Works” > “perfect ontology engineering”
- Safe defaults: assertions go into a session microtheory (not BaseKB), except for creating the session microtheory itself.

---

## Directory layout

- `java-server/` : HTTP wrapper around `org.opencyc.api.CycAccess`
- `python-cli/`  : CLI orchestrator + Ollama client

---

## Quick start

### 0) Prerequisites

- Java 11+
- Python 3.10+
- OpenCyc running locally (default API port is typically `3601`; web UI is typically `3602`)
- Ollama installed and a model pulled (e.g. `ollama pull llama3.2` or `ollama pull deepseek-r1`)

---

## 1) Build & run the Java OpenCyc HTTP bridge

### 1.1 Provide the OpenCyc API jar

The OpenCyc Java API jar is not in Maven Central.

Copy your OpenCyc API jar into:

```
java-server/lib/opencyc-api.jar
```

Notes:
- In some OpenCyc distributions the jar is named `OpenCyc.jar` or similar; you can rename it to `opencyc-api.jar`.

### 1.2 Build

From `java-server/`:

```bash
mvn -q -DskipTests package
```

This produces:

```
java-server/target/cyc-bridge-server.jar
```

### 1.3 Run

```bash
export CYC_HOST=localhost
export CYC_PORT=3601
export CYC_BRIDGE_HTTP_PORT=8081

java -jar target/cyc-bridge-server.jar
```

Health check:

```bash
curl http://localhost:8081/health
```

---

## 2) Run the Python CLI

### 2.1 Install deps

From `python-cli/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Configure

You can use environment variables:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.2
export CYC_BRIDGE_BASE_URL=http://localhost:8081
```

### 2.3 Start interactive CLI

```bash
python -m cyc_llm_cli
```

Type questions at the prompt. Use `/help` for commands.

---

## CLI commands

- `/help`      show help
- `/debug on`  show raw plan + Cyc calls
- `/debug off`
- `/session`   show current session microtheory
- `/exit`      quit

---

## Security notes

- This is intended to run **locally**.
- The HTTP bridge includes a `/api/v1/converse` endpoint to run arbitrary SubL. Keep it local/firewalled.

---

## Extending next

Typical next steps:
- richer Cyc term lookup (nameString, synonyms, FI-complete)
- multi-variable queries (askWithVariables / Cyc queries)
- proof/provenance capture
- a web UI on top of the same HTTP bridge
