# SRE-Agent-Sandbox

> OpenEnv-compliant RL environment simulating a distributed 3-tier microservices system under stress. An AI agent acts as an SRE, diagnosing and remediating infrastructure failures in real time.

Built for the **Meta PyTorch OpenEnv Hackathon**.

## Features

- **3-Tier Simulated System** — API Gateway → Order Service → DB/Cache with realistic metrics (CPU, memory, latency, request count), dependency chains, and cascading failures
- **Chaos Engine** — Probabilistic fault injection: memory leaks (gradual), latent dependency timeouts (cascade upstream), bad configs (immediate 500s)
- **5-Component Reward Function** — Availability, latency penalty, downtime penalty, efficiency penalty, safety penalty
- **OpenEnv Compliant** — `reset()` / `step()` / `state` API with `SREAction(Action)`, `SREObservation(Observation)`, `SREState(State)`
- **FastAPI Server** — REST + WebSocket endpoints for agent communication
- **Docker Ready** — Multi-stage build, < 500 MB image, single command to run
- **Deterministic Replay** — Seeded randomness for reproducible training episodes
- **ASCII Dashboard** — Terminal-rendered system status for debugging and demos

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd sre-agent-sandbox
uv sync

# Run the demo (random agent vs heuristic agent)
uv run python -m sre_agent_sandbox.demo.run_demo

# Start the server
uv run uvicorn sre_agent_sandbox.server.app:app --host 0.0.0.0 --port 8000
```

### Interact via curl

```bash
# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}' -c cookies.txt

# Take an action (RestartService on db)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": 1, "target_service": "db"}}' -b cookies.txt

# Get current state
curl http://localhost:8000/state -b cookies.txt

# Health check
curl http://localhost:8000/health
```

## Architecture

```
sre-agent-sandbox/
├── openenv.yaml                 # OpenEnv environment descriptor
├── pyproject.toml               # Project metadata & dependencies
├── Dockerfile                   # Multi-stage Docker build
├── sre_agent_sandbox/
│   ├── models.py                # SREAction, SREObservation, SREState (Pydantic)
│   ├── simulated_system.py      # 3-tier system state model
│   ├── chaos_engine.py          # Probabilistic fault injection
│   ├── reward.py                # 5-component reward calculator
│   ├── renderer.py              # ASCII terminal dashboard
│   ├── server/
│   │   ├── environment.py       # SREEnvironment (OpenEnv Environment subclass)
│   │   └── app.py               # FastAPI REST + WebSocket server
│   └── demo/
│       └── run_demo.py          # Random & heuristic demo agents
├── demo/
│   └── run_demo.py              # Top-level demo entry point
└── tests/                       # Comprehensive test suite
    ├── test_models.py
    ├── test_simulated_system.py
    ├── test_chaos_engine.py
    ├── test_reward.py
    ├── test_environment.py
    ├── test_server.py
    ├── test_renderer.py
    └── test_demo.py
```

## Action Space

The agent selects an **action type** and a **target service** each step:

| Action Type | Name           | Effect                                              |
|:-----------:|----------------|------------------------------------------------------|
| 0           | NoOp           | Do nothing                                           |
| 1           | RestartService | Reset target metrics to baseline (clears all faults) |
| 2           | Rollback       | Revert to stable config (clears bad_config fault)    |
| 3           | ScaleUp        | Add instance, reduce CPU & latency proportionally    |
| 4           | ClearCache     | Reset memory & latency to baseline                   |

**Target services:** `api`, `order`, `db`

```python
from sre_agent_sandbox.models import SREAction

action = SREAction(action_type=1, target_service="db")  # Restart DB
```

## Observation Space

Each step returns an `SREObservation` with:

| Field           | Type                         | Description                                      |
|-----------------|------------------------------|--------------------------------------------------|
| `metrics`       | `dict[str, dict[str, float]]`| Per-service: `{cpu, memory, latency, request_count}` |
| `log_buffer`    | `list[str]`                  | Last 10 log entries (FIFO)                       |
| `health_status` | `dict[str, bool]`            | Per-service health: `{api, order, db}`           |
| `active_alerts` | `list[str]`                  | Current alert messages                           |
| `reward`        | `float \| None`              | Step reward (None on reset)                      |
| `done`          | `bool`                       | Whether the episode has ended                    |

## Reward Function

The reward is computed as the sum of 5 components:

| Component     | Value                                        | Condition                            |
|---------------|----------------------------------------------|--------------------------------------|
| Availability  | **+1.0**                                     | All services healthy                 |
| Latency       | **−0.01 × (avg_latency − 100)**             | When avg latency > 100ms            |
| Downtime      | **−5.0**                                     | Any service is down                  |
| Efficiency    | **−0.1**                                     | Any non-NoOp action taken            |
| Safety        | **−10.0**                                    | Restarting a healthy service         |

The agent must balance fixing failures quickly (availability, downtime) against unnecessary interventions (efficiency, safety).

## Chaos Engine — Fault Types

| Fault              | Behaviour                                                              |
|--------------------|------------------------------------------------------------------------|
| `memory_leak`      | Memory grows +8% per tick; service crashes (DOWN) at 95%              |
| `latent_dependency`| Latency grows +20ms/tick on target, cascades upstream at 60% factor   |
| `bad_config`       | Immediate: service marked unhealthy, requires Rollback to fix          |

Faults are injected probabilistically each step (default 30% chance). The dependency chain `db → order → api` means a DB latency fault cascades through the entire stack.

## Termination Conditions

An episode ends when:
- **Max steps reached** (default: 200) — the agent survived
- **Total meltdown** — all 3 services are simultaneously down

## API Reference

### REST Endpoints

| Method | Path      | Description                            |
|--------|-----------|----------------------------------------|
| GET    | `/health` | Liveness probe (`{"status": "ok"}`)    |
| GET    | `/schema` | JSON schemas for action & observation  |
| POST   | `/reset`  | Reset environment, returns observation |
| POST   | `/step`   | Take action, returns step result       |
| GET    | `/state`  | Current SREState                       |

**POST /reset** body:
```json
{"seed": 42, "episode_id": "my-episode"}
```

**POST /step** body:
```json
{"action": {"action_type": 1, "target_service": "db"}}
```

**POST /step** response:
```json
{
  "observation": {
    "metrics": {"api": {"cpu": 30.0, "memory": 40.0, "latency": 50.0, "request_count": 100.0}, "...": "..."},
    "log_buffer": ["RestartService applied to db: metrics reset to baseline"],
    "health_status": {"api": true, "order": true, "db": true},
    "active_alerts": []
  },
  "reward": 0.9,
  "done": false,
  "terminated": false,
  "truncated": false,
  "info": {}
}
```

### WebSocket Protocol

Connect to `ws://localhost:8000/ws`. Send JSON messages:

```jsonc
// Reset
{"type": "reset", "data": {"seed": 42}}

// Step
{"type": "step", "data": {"action": {"action_type": 2, "target_service": "order"}}}

// Get state
{"type": "state"}

// Close connection
{"type": "close"}
```

Response types: `observation`, `step_result`, `state`, `error`.

## Demo Agents

The demo runs two agents side-by-side for comparison:

**Random Agent** — selects random actions each step (baseline).

**Heuristic Agent** — rule-based policy:
1. `bad_config` alert → **Rollback** on affected service
2. `memory_leak` alert → **RestartService** on affected service
3. `timeout`/`latency` alert → **RestartService** on root cause service
4. High CPU (>80%) with no faults → **ScaleUp**
5. All healthy → **NoOp**

```bash
# Run with default settings (5 episodes each, ASCII dashboard)
uv run python -m sre_agent_sandbox.demo.run_demo

# Or via the project script entry point
uv run sre-agent-demo
```

## Docker Usage

```bash
# Build the image
docker build -t sre-agent-sandbox .

# Run the server
docker run -p 8000:8000 sre-agent-sandbox

# Verify
curl http://localhost:8000/health
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_environment.py

# Run with coverage (if pytest-cov is installed)
uv run pytest --cov=sre_agent_sandbox
```

## Programmatic Usage

```python
from sre_agent_sandbox.server.environment import SREEnvironment
from sre_agent_sandbox.models import SREAction

# Create environment
env = SREEnvironment(max_steps=200, fault_probability=0.3)

# Reset with a seed for reproducibility
obs = env.reset(seed=42)

# Run an episode
done = False
total_reward = 0.0
while not done:
    # Your agent logic here
    action = SREAction(action_type=0, target_service="api")
    obs = env.step(action)
    total_reward += obs.reward or 0.0
    done = obs.done

print(f"Episode finished — Total reward: {total_reward:.3f}")
print(f"State: {env.state}")
```

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager
- Dependencies: `openenv-core`, `fastapi`, `uvicorn`, `pydantic`
- Dev: `pytest`, `httpx`, `websockets`, `ruff`

## License

Built for the Meta PyTorch OpenEnv Hackathon.
