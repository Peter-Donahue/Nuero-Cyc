# Java OpenCyc HTTP Bridge

This is a minimal HTTP/JSON wrapper around the legacy OpenCyc Java API (`org.opencyc.api.CycAccess`).

## Build

1) Copy your OpenCyc Java API jar to:

```
lib/opencyc-api.jar
```

2) Build with Maven:

```bash
mvn -q -DskipTests package
```

3) Run:

```bash
export CYC_HOST=localhost
export CYC_PORT=3601
export CYC_BRIDGE_HTTP_PORT=8081
java -jar target/cyc-bridge-server.jar
```

## Endpoints

- `GET /health`
- `POST /api/v1/session`
- `POST /api/v1/constant/exists`
- `POST /api/v1/constant/create`
- `POST /api/v1/assert`
- `POST /api/v1/ask_true`
- `POST /api/v1/ask_var`
- `POST /api/v1/converse` (dangerous; keep local)
