# SRE-Agent-Sandbox

> OpenEnv-compliant RL environment simulating a distributed 3-tier microservices system under stress. An AI agent acts as an SRE, diagnosing and remediating infrastructure failures in real time.

Built for the **Meta PyTorch OpenEnv Hackathon**.

## Motivation

Site Reliability Engineering (SRE) is a high-stakes, real-world discipline where engineers must rapidly diagnose cascading failures, choose the right remediation strategy, and balance speed against safety. This environment captures that challenge in a reproducible RL setting:

- A **3-tier microservices system** (API Gateway, Order Service, DB/Cache) with realistic dependency chains where a database failure cascades through the entire stack.
- A **chaos engine** that probabilistically injects faults (memory leaks, latent dependency timeouts, bad configurations), forcing the agent to triage under uncertainty.
- A **penalty-heavy reward function** that mirrors real-world incentives: keeping services healthy earns small credit, but every failure and unnecessary intervention costs heavily.

The result is an environment where an agent must learn to diagnose fault types from noisy observations, pick the correct remediation action for each fault, and avoid wasteful or dangerous actions -- just like a real SRE.

## Features

- **3-Tier Simulated System** -- API Gateway -> Order Service -> DB/Cache with realistic metrics (CPU, memory, latency, request count), dependency chains, and cascading failures
- **Chaos Engine** -- Probabilistic fault injection: memory leaks (gradual), latent dependency timeouts (cascade upstream), bad configs (immediate)
- **5-Component Reward Function** -- Availability, latency penalty, downtime penalty, efficiency penalty, safety penalty
- **3 Difficulty Levels** -- Easy, medium, and hard tasks with graders scoring 0.0-1.0
- **OpenEnv Compliant** -- `reset()` / `step()` / `state` API with typed `SREAction`, `SREObservation`, `SREState`
- **FastAPI Server** -- REST + WebSocket endpoints for remote agent communication
- **Docker Ready** -- Multi-stage build, single command to run
- **Deterministic Replay** -- Seeded randomness for reproducible training and evaluation
- **ASCII Dashboard** -- Terminal-rendered system status for debugging and demos

## Setup

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager

### Install and Run

```bash
# Clone and install
git clone <repo-url> && cd sre-agent-sandbox
uv sync

# Run baseline evaluation (all 3 tasks, graded 0.0-1.0)
uv run python -m sre_agent_sandbox.baseline_eval

# Run the interactive demo (random agent vs heuristic agent with ASCII dashboard)
uv run python -m sre_agent_sandbox.demo.run_demo

# Start the HTTP server for remote agents
uv run uvicorn sre_agent_sandbox.server.app:app --host 0.0.0.0 --port 8000

# Run tests
uv run pytest
```

### Docker

```bash
docker build -t sre-agent-sandbox .
docker run -p 8000:8000 sre-agent-sandbox
curl http://localhost:8000/health
```

---

## Action Space

Each step, the agent selects an **action type** and a **target service**:

```json
{"action_type": 1, "target_service": "db"}
```

| Type | Name             | Effect                                                    |
|:----:|------------------|-----------------------------------------------------------|
| 0    | **NoOp**         | Do nothing (no cost if system is healthy)                 |
| 1    | **RestartService**| Reset target metrics to baseline, clears all faults       |
| 2    | **Rollback**     | Revert to stable config, clears `bad_config` fault only   |
| 3    | **ScaleUp**      | Add instance, reduce CPU and latency proportionally       |
| 4    | **ClearCache**   | Reset memory and latency to baseline                      |

**Target services:** `api`, `order`, `db`

Total action space: **5 actions x 3 services = 15 discrete actions**.

```python
from sre_agent_sandbox.models import SREAction
action = SREAction(action_type=2, target_service="db")  # Rollback DB
```

## Observation Space

Each step returns an `SREObservation` with:

| Field            | Type                          | Description                                             |
|------------------|-------------------------------|---------------------------------------------------------|
| `metrics`        | `dict[str, dict[str, float]]` | Per-service: `{cpu, memory, latency, request_count}`    |
| `log_buffer`     | `list[str]`                   | Last 10 log entries (FIFO)                              |
| `health_status`  | `dict[str, bool]`             | Per-service health: `{api, order, db}`                  |
| `active_alerts`  | `list[str]`                   | Current alert messages (fault type, service, timeouts)  |
| `reward`         | `float | None`                | Step reward (`None` on reset)                           |
| `done`           | `bool`                        | Whether the episode has ended                           |

**Example observation (JSON):**

```json
{
  "metrics": {
    "api": {"cpu": 32.1, "memory": 41.5, "latency": 55.0, "request_count": 98},
    "order": {"cpu": 30.0, "memory": 40.0, "latency": 50.0, "request_count": 100},
    "db": {"cpu": 85.0, "memory": 90.0, "latency": 320.0, "request_count": 45}
  },
  "health_status": {"api": true, "order": true, "db": false},
  "active_alerts": ["Active fault on db: bad_config", "db is UNHEALTHY"],
  "log_buffer": ["BadConfig injected on db: service immediately unhealthy"],
  "reward": -5.1,
  "done": false
}
```

## State

The `SREState` (accessed via `env.state`) provides episode metadata:

| Field                | Type            | Description                                  |
|----------------------|-----------------|----------------------------------------------|
| `episode_id`         | `str | None`    | Unique episode identifier                    |
| `step_count`         | `int`           | Current step number                          |
| `system_health_score`| `float` (0-1)   | Fraction of services currently healthy       |
| `active_incidents`   | `list[str]`     | Active fault descriptions                    |

---

## Reward Function

The reward is the sum of 5 components, computed **after** chaos injection each step:

| Component          | Value                          | When                                |
|--------------------|--------------------------------|-------------------------------------|
| **Availability**   | **+1.0**                       | All 3 services healthy              |
| **Latency**        | **-0.01 x (avg_lat - 100)**    | Average latency > 100ms            |
| **Downtime**       | **-5.0**                       | Any service is down                 |
| **Efficiency**     | **-0.1**                       | Any non-NoOp action taken           |
| **Safety**         | **-10.0**                      | Restarting a healthy service        |

- **Best step:** +1.0 (all healthy, NoOp)
- **Typical good fix:** +0.9 (all healthy after remediation, -0.1 efficiency cost)
- **Typical fault step:** -5.1 (one service down, agent acts)
- **Worst step:** -15.1 (service down + restart healthy service)

Rewards are intentionally **penalty-heavy**: the agent can only react after faults are injected, so even a perfect policy accumulates negative reward from the unavoidable reaction delay. This creates a meaningful signal where the score reflects **how quickly and correctly** the agent remediates.

### Determinism

The reward function is **fully deterministic and stateless**. `RewardCalculator` takes the current system state and the action as inputs and returns the same reward every time -- no randomness, no learned parameters, no history dependence. Each component is a pure function of observable state (health flags, latency values, service up/down status) and the action taken. The stochasticity in the environment comes from the **chaos engine** (fault injection) and **metric drift**, both of which use a seeded `random.Random(seed)`. Fixing the seed makes the entire episode -- including all reward values -- fully reproducible.

## Chaos Engine -- Fault Types

| Fault                | Behaviour                                                             | Correct Remediation     |
|----------------------|-----------------------------------------------------------------------|-------------------------|
| `memory_leak`        | Memory grows +8%/tick; service crashes (DOWN) at 95%                 | RestartService or ClearCache |
| `latent_dependency`  | Latency grows +20ms/tick on target, cascades upstream at 60% factor  | RestartService          |
| `bad_config`         | Immediate: service marked unhealthy, requires config revert          | Rollback                |

The dependency chain `db -> order -> api` means a DB latency fault cascades through the entire stack.

## Termination Conditions

An episode ends when:
- **Max steps reached** -- the agent survived the full episode
- **Total meltdown** -- all 3 services are simultaneously down

---

## Tasks

The environment defines **3 tasks** with increasing difficulty. Each task uses fixed evaluation seeds for reproducible grading. Scores are normalized to **0.0-1.0** using calibrated reward bounds.

### Task 1: Single Fault Remediation (Easy)

| Parameter           | Value        |
|---------------------|--------------|
| Task ID             | `sre_single_fault` |
| Fault types         | `bad_config` only |
| Fault probability   | 10%          |
| Max steps           | 100          |
| Eval episodes       | 5 (seeds: 100-104) |

**What the agent must learn:** Detect `bad_config` alerts and rollback the affected service. A single correct policy (if bad_config -> rollback) is sufficient. This task tests whether the agent can map observations to the right action.

### Task 2: Mixed Fault Diagnosis (Medium)

| Parameter           | Value        |
|---------------------|--------------|
| Task ID             | `sre_mixed_faults` |
| Fault types         | All 3 (memory_leak, latent_dependency, bad_config) |
| Fault probability   | 30%          |
| Max steps           | 200          |
| Eval episodes       | 5 (seeds: 200-204) |

**What the agent must learn:** Diagnose which fault type is active from observation signals (memory climbing vs latency spiking vs unhealthy flag) and choose the correct remediation for each. Wrong actions waste steps and accumulate penalties.

### Task 3: High Chaos Survival (Hard)

| Parameter           | Value        |
|---------------------|--------------|
| Task ID             | `sre_high_chaos` |
| Fault types         | All 3        |
| Fault probability   | 50%          |
| Latency timeout     | 350ms (tighter than default 500ms) |
| Max steps           | 300          |
| Eval episodes       | 5 (seeds: 300-304) |

**What the agent must learn:** Rapid triage under relentless chaos. Multiple concurrent faults across different services, cascading latency hitting tighter thresholds, and the constant risk of total meltdown. The agent must prioritize which fault to fix first and tolerate some degradation while handling the most critical issues.

---

## Baseline Scores

### Programmatic Baselines

Scores from `uv run python -m sre_agent_sandbox.baseline_eval` (deterministic, reproducible):

| Task                  | Difficulty | Random Agent | Heuristic Agent |
|-----------------------|------------|:------------:|:---------------:|
| Single Fault          | Easy       | 0.37         | **0.97**        |
| Mixed Fault Diagnosis | Medium     | 0.20         | **0.70**        |
| High Chaos Survival   | Hard       | 0.57         | **0.46**        |

**Interpreting the scores:**
- **Easy:** The heuristic nearly solves it (0.97) because a single if/else rule suffices. A trained agent should reach 0.95+.
- **Medium:** The heuristic gets 0.70 by applying correct remediations per fault type. Room for a trained agent to exceed this with better timing and prioritization.
- **Hard:** Even the heuristic breaks down (0.46) under 50% fault rate with tighter thresholds. The random agent's higher score (0.57) reflects lucky early terminations from meltdown on bad seeds. This task requires sophisticated prioritization that simple rules cannot achieve.

### LLM Baseline Inference

The LLM baseline uses the OpenAI API to run a language model as the agent:

```bash
# Install the optional LLM dependency
uv pip install openai

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run inference (defaults to gpt-4o-mini)
uv run python -m sre_agent_sandbox.baseline_inference

# Use a different model
OPENAI_MODEL="gpt-4o" uv run python -m sre_agent_sandbox.baseline_inference
```

The script reads `OPENAI_API_KEY` from environment variables, runs the model against all 3 tasks using the same deterministic evaluation seeds, and produces graded 0.0-1.0 scores.

**Grading formula:**

```
score = clamp((cumulative_reward - worst) / (best - worst), 0.0, 1.0)
```

Where `worst` and `best` are calibrated per-task reward bounds (see `tasks.py`).

---

## API Reference

### REST Endpoints

| Method | Path      | Description                            |
|--------|-----------|----------------------------------------|
| GET    | `/health` | Liveness probe (`{"status": "healthy"}`)|
| GET    | `/schema` | JSON schemas for action and observation|
| POST   | `/reset`  | Reset environment, returns observation |
| POST   | `/step`   | Take action, returns step result       |
| GET    | `/state`  | Current SREState                       |

```bash
# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}' -c cookies.txt

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": 1, "target_service": "db"}}' -b cookies.txt

# State
curl http://localhost:8000/state -b cookies.txt
```

### WebSocket

Connect to `ws://localhost:8000/ws` and send JSON:

```json
{"type": "reset", "data": {"seed": 42}}
{"type": "step", "data": {"action": {"action_type": 2, "target_service": "order"}}}
{"type": "state"}
```

---

## Programmatic Usage

```python
from sre_agent_sandbox.server.environment import SREEnvironment
from sre_agent_sandbox.models import SREAction

env = SREEnvironment(max_steps=200, fault_probability=0.3)
obs = env.reset(seed=42)

done = False
while not done:
    action = SREAction(action_type=0, target_service="api")  # your agent here
    obs = env.step(action)
    done = obs.done

print(f"Reward: {env._cumulative_reward:.3f}")
```

### Task-Aware Evaluation

```python
from sre_agent_sandbox.tasks import TASKS, evaluate_agent

task = TASKS["sre_mixed_faults"]
result = evaluate_agent(task, your_agent)
print(f"Score: {result['mean_score']:.3f}")  # 0.0-1.0
```

---

## Architecture

```
sre-agent-sandbox/
├── openenv.yaml                 # OpenEnv manifest with task definitions
├── pyproject.toml               # Project metadata and dependencies
├── Dockerfile                   # Multi-stage Docker build
├── server/
│   └── app.py                   # Top-level server entry (openenv validate)
├── sre_agent_sandbox/
│   ├── models.py                # SREAction, SREObservation, SREState (Pydantic)
│   ├── simulated_system.py      # 3-tier system state model
│   ├── chaos_engine.py          # Probabilistic fault injection
│   ├── reward.py                # 5-component reward calculator
│   ├── tasks.py                 # Task definitions (easy/medium/hard) and graders
│   ├── baseline_eval.py         # Programmatic baseline evaluation
│   ├── baseline_inference.py    # LLM baseline using OpenAI API
│   ├── renderer.py              # ASCII terminal dashboard
│   ├── server/
│   │   ├── environment.py       # SREEnvironment (OpenEnv Environment subclass)
│   │   └── app.py               # FastAPI REST + WebSocket server
│   └── demo/
│       └── run_demo.py          # Random and heuristic demo agents
└── tests/
    ├── test_models.py
    ├── test_simulated_system.py
    ├── test_chaos_engine.py
    ├── test_reward.py
    ├── test_environment.py
    ├── test_server.py
    ├── test_renderer.py
    ├── test_demo.py
    └── test_tasks.py
```

## License

Built for the Meta PyTorch OpenEnv Hackathon.
