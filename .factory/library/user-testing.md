# User Testing

Testing surface, resource cost classification, and validation approach.

---

## Validation Surface
- **Terminal/CLI**: Run demo script, observe ASCII dashboard output
- **API endpoints**: curl to REST endpoints (reset, step, state, health, schema)
- **Docker**: Build image, run container, verify connectivity
- No browser needed — all validation is API/terminal based

## Validation Tools
- pytest (unit + integration tests)
- curl (REST endpoint testing)
- Python websockets library (WebSocket testing)
- docker CLI (build/run/stop)
- Prefer `uv run ...` (or `.venv/bin/python -m ...`) for Python tooling; system `python3` may not have project test dependencies (for example, `pytest`).

## Validation Concurrency
- **Max concurrent validators: 5**
- Rationale: Each validator is a lightweight Python process (~100MB RAM). Machine has 16GB RAM, 10 cores, ~6GB baseline usage. Headroom: 10GB * 0.7 = 7GB. 5 validators = ~500MB — well within budget.
- Docker tests should be serialized (single container on port 8000)

## Known Constraints
- Docker must be running for Docker-related validation
- Port 8000 must be free for server/Docker tests
- Environment step() is purely in-memory, no external services
- In current core environment behavior, `bad_config` fault injection immediately flips health but does not by itself increase latency; for metric-restoration checks, establish a degraded metric state before remediation.
- REST `/step` endpoint expects payload shape `{"action": {"action_type": <int>, "target_service": "<svc>"}}`; sending raw action fields returns HTTP 422.
- REST session state is cookie-backed; clients must preserve cookies across `/reset` -> `/step` -> `/state` requests or `/step` may return HTTP 409 (not reset).

## Flow Validator Guidance: terminal-cli
- Surface scope: Python/CLI validation only (`pytest`, short Python scripts, optional `curl` against local services when explicitly required by assigned assertions).
- Isolation boundary: stay inside `/Users/anant/ai/sre-agent-sandbox` and assigned assertion set; do not modify application code.
- Allowed writes: your assigned flow report at `.factory/validation/<milestone>/user-testing/flows/<group-id>.json` and evidence artifacts under `<missionDir>/evidence/<milestone>/<group-id>/`.
- Concurrency safety: avoid starting background services unless your assertion group requires it; if started, stop them before exit. Do not use ports other than 8000.
- Evidence expectations: record exact commands, exit codes, and key observations for each assertion mapping.
