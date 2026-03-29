---
name: server-worker
description: Implements FastAPI server, Docker, ASCII visualization, and demo client
---

# Server Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features involving:
- FastAPI server (REST endpoints, WebSocket)
- Dockerfile and Docker build
- ASCII render() visualization
- Demo client scripts (random/heuristic agents)
- Integration tests for server endpoints
- openenv.yaml updates

## Work Procedure

1. **Read context**: Read `AGENTS.md`, `.factory/research/openenv-api.md`, `.factory/library/architecture.md`, and `.factory/services.yaml` to understand the API patterns and service configuration.

2. **Write tests first (TDD)**:
   - For server features: use `httpx.AsyncClient` with `TestClient` from FastAPI
   - For WebSocket: use `websockets` library in test
   - For Docker: tests can verify Dockerfile exists and syntax
   - For demo: tests verify script runs and produces expected output
   - Run `uv run pytest tests/ -v --tb=short -x` — tests should FAIL (red phase)

3. **Implement the feature**:
   - **Server**: FastAPI app with POST /reset, POST /step, GET /state, GET /health, GET /schema, WS /ws
   - **Docker**: Two-stage Dockerfile (uv builder + python:3.12-slim), expose 8000, CMD uvicorn
   - **Render**: ASCII dashboard function outputting service grid, metrics, alerts, reward. Max 120 chars wide.
   - **Demo**: Script with random agent + heuristic agent (rollback for bad-config, restart for others)
   - Import environment modules from `sre_agent_sandbox.*`

4. **Run tests (green phase)**:
   - `uv run pytest tests/ -v --tb=short -x` — all tests must pass

5. **Run full test suite**:
   - `uv run pytest tests/ -v --tb=short` — verify no regressions

6. **Manual verification**:
   - **Server**: Start server (`uv run uvicorn sre_agent_sandbox.server.app:app --port 8000`), curl /health, /reset, /step, /state. Then stop it.
   - **Docker**: `docker build -t sre-agent-sandbox .`, `docker run -d -p 8000:8000 sre-agent-sandbox`, curl /health, then `docker stop`. Check image size.
   - **Render**: Call render() and inspect ASCII output in terminal
   - **Demo**: Run demo script, verify both agents complete episodes, check reward comparison
   - IMPORTANT: Always stop any server/container you start. Kill processes on port 8000 when done.

7. **Update library** if you discover patterns: `.factory/library/architecture.md`

## Example Handoff

```json
{
  "salientSummary": "Implemented FastAPI server with all OpenEnv endpoints (POST /reset, POST /step, GET /state, GET /health, GET /schema, WS /ws). Ran `uv run pytest tests/test_server.py -v` (15 passing) and manually verified all endpoints via curl after starting the server on port 8000.",
  "whatWasImplemented": "FastAPI application in sre_agent_sandbox/server/app.py with REST endpoints (reset returns observation, step accepts action and returns observation+reward+done, state returns current SREState, health returns {status: ok}, schema returns JSON schemas for action/observation). WebSocket endpoint at /ws supporting reset/step/state/close message types with error handling for malformed messages. Session management via dependency injection.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv run pytest tests/test_server.py -v --tb=short", "exitCode": 0, "observation": "15 tests passed: health endpoint, schema endpoint, reset, step, step-before-reset, state, websocket lifecycle, websocket error handling"},
      {"command": "curl -s http://localhost:8000/health", "exitCode": 0, "observation": "{\"status\":\"ok\"}"},
      {"command": "curl -s -X POST http://localhost:8000/reset | python3 -m json.tool | head -20", "exitCode": 0, "observation": "Valid observation JSON with metrics, health_status, log_buffer, done=false"},
      {"command": "lsof -ti :8000 | xargs kill", "exitCode": 0, "observation": "Server stopped cleanly"}
    ],
    "interactiveChecks": [
      {"action": "Started uvicorn server on port 8000", "observed": "Server started, health endpoint returned 200"},
      {"action": "POST /reset then POST /step with valid action", "observed": "Both returned valid JSON responses with correct structure"},
      {"action": "POST /step without prior reset", "observed": "Returned 409 with error message"},
      {"action": "Stopped server on port 8000", "observed": "Process killed cleanly, port freed"}
    ]
  },
  "tests": {
    "added": [
      {"file": "tests/test_server.py", "cases": [
        {"name": "test_health_endpoint", "verifies": "GET /health returns 200 with status ok"},
        {"name": "test_reset_returns_observation", "verifies": "POST /reset returns valid SREObservation"},
        {"name": "test_step_before_reset_rejected", "verifies": "POST /step without reset returns 4xx"},
        {"name": "test_websocket_lifecycle", "verifies": "Full WS session: reset->step->state->close"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Core environment modules (models, chaos_engine, etc.) are missing or broken
- Docker daemon is not running
- Port 8000 is occupied by another process
- OpenEnv server creation patterns don't work as documented
