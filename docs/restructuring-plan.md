# SRE Agent Sandbox — Restructuring Plan

## 1. Current File Inventory with Import Graph

### 1.1 Root-Level Shim Files (CURRENT — to become REAL files)

| File | Imports FROM | Imported BY |
|---|---|---|
| `__init__.py` | `sre_agent_sandbox.models` → SREAction, SREObservation, SREState | (package init) |
| `models.py` | `sre_agent_sandbox.models` → SREAction, SREObservation, SREState | (none directly) |
| `client.py` | `openenv.core.EnvClient`, `openenv.core.client_types.StepResult`, `openenv.core.env_server.types.State`, `sre_agent_sandbox.models` → SREAction, SREObservation, SREState | (none directly) |
| `server/__init__.py` | (empty) | — |
| `server/app.py` | `openenv.core.env_server.http_server.create_app`, `sre_agent_sandbox.models` → SREAction, SREObservation, `sre_agent_sandbox.server.environment` → SREEnvironment | `pyproject.toml` script: `server = "server.app:main"` |

### 1.2 Inner Package Files (sre_agent_sandbox/ — REAL code, to be eliminated as a nested package)

| File | Imports FROM | Imported BY |
|---|---|---|
| `sre_agent_sandbox/__init__.py` | (empty docstring only) | — |
| `sre_agent_sandbox/models.py` | `openenv.core.env_server.types` → Action, Observation, State; `pydantic` → Field, field_validator | Root `__init__.py`, root `models.py`, root `client.py`, root `server/app.py`, inner `server/app.py`, inner `server/environment.py`, inner `reward.py`, inner `baseline_inference.py`, inner `demo/run_demo.py`, ALL test files |
| `sre_agent_sandbox/simulated_system.py` | `random` (stdlib only) | `chaos_engine.py`, `reward.py`, `renderer.py`, `server/environment.py`, tests: `test_chaos_engine.py`, `test_simulated_system.py`, `test_reward.py`, `test_environment.py`, `test_renderer.py` |
| `sre_agent_sandbox/chaos_engine.py` | `sre_agent_sandbox.simulated_system` → DEPENDENCIES, SERVICE_NAMES, SimulatedSystem | `server/environment.py`, tests: `test_chaos_engine.py` |
| `sre_agent_sandbox/reward.py` | `sre_agent_sandbox.models` → SREAction; `sre_agent_sandbox.simulated_system` → SERVICE_NAMES, SimulatedSystem | `server/environment.py`, tests: `test_reward.py` |
| `sre_agent_sandbox/renderer.py` | `sre_agent_sandbox.simulated_system` → SERVICE_NAMES; TYPE_CHECKING: `sre_agent_sandbox.server.environment` → SREEnvironment | `demo/run_demo.py` (imported at call site), `tasks.py`, tests: `test_renderer.py` |
| `sre_agent_sandbox/tasks.py` | `sre_agent_sandbox.server.environment` → SREEnvironment; `sre_agent_sandbox.renderer` → render | `baseline_eval.py`, `baseline_inference.py`, tests: `test_tasks.py` |
| `sre_agent_sandbox/baseline_eval.py` | `sre_agent_sandbox.demo.run_demo` → HeuristicAgent, RandomAgent; `sre_agent_sandbox.tasks` → TASKS, evaluate_agent | (script entry point) |
| `sre_agent_sandbox/baseline_inference.py` | `sre_agent_sandbox.models` → SREAction, SREObservation, SREState; `sre_agent_sandbox.tasks` → TASKS, _configure_env_for_task, grade, make_env | (script entry point) |
| `sre_agent_sandbox/server/__init__.py` | (docstring only) | — |
| `sre_agent_sandbox/server/app.py` | `openenv.core.env_server.http_server` → create_app; `sre_agent_sandbox.models` → SREAction, SREObservation; `sre_agent_sandbox.server.environment` → SREEnvironment | Dockerfile CMD, `test_server.py` |
| `sre_agent_sandbox/server/environment.py` | `openenv.core.env_server.interfaces` → Environment; `sre_agent_sandbox.chaos_engine` → ChaosEngine; `sre_agent_sandbox.models` → SREAction, SREObservation, SREState; `sre_agent_sandbox.reward` → RewardCalculator; `sre_agent_sandbox.simulated_system` → SERVICE_NAMES, SimulatedSystem | `server/app.py`, `tasks.py`, `renderer.py` (TYPE_CHECKING), `demo/run_demo.py`, tests: `test_environment.py`, `test_renderer.py`, `test_demo.py` |
| `sre_agent_sandbox/demo/__init__.py` | (docstring only) | — |
| `sre_agent_sandbox/demo/run_demo.py` | `sre_agent_sandbox.models` → SREAction, SREObservation, SREState; `sre_agent_sandbox.server.environment` → SREEnvironment; `sre_agent_sandbox.simulated_system` → SERVICE_NAMES; `sre_agent_sandbox.renderer` → render (lazy import) | `baseline_eval.py`, tests: `test_demo.py`, `test_tasks.py` |
| `sre_agent_sandbox/server/Dockerfile` | N/A (build file referencing `sre_agent_sandbox/` in COPY) | — |

### 1.3 Root-Level Non-Code Files

| File | Notes |
|---|---|
| `pyproject.toml` | Scripts: `server = "server.app:main"`, `sre-agent-demo = "sre_agent_sandbox.demo.run_demo:main"`. `pythonpath = ["."]` |
| `openenv.yaml` | `entry_point: sre_agent_sandbox.server.environment:SREEnvironment`; `grader: sre_agent_sandbox.tasks:grade` |
| `Dockerfile` | COPYs `sre_agent_sandbox/` dir; CMD runs `sre_agent_sandbox.server.app:app` |
| `demo/__init__.py` | Docstring only |
| `demo/run_demo.py` | Shim: imports `sre_agent_sandbox.demo.run_demo.main` |

### 1.4 Test Files Import Summary

| Test File | Imports |
|---|---|
| `test_models.py` | `sre_agent_sandbox.models` → SREAction, SREObservation, SREState |
| `test_simulated_system.py` | `sre_agent_sandbox.simulated_system` → SimulatedSystem |
| `test_chaos_engine.py` | `sre_agent_sandbox.chaos_engine` → ChaosEngine; `sre_agent_sandbox.simulated_system` → BASELINE_METRICS, SERVICE_NAMES, SimulatedSystem |
| `test_reward.py` | `sre_agent_sandbox.models` → SREAction; `sre_agent_sandbox.reward` → RewardCalculator; `sre_agent_sandbox.simulated_system` → SimulatedSystem |
| `test_environment.py` | `sre_agent_sandbox.models` → SREAction, SREObservation, SREState; `sre_agent_sandbox.server.environment` → SREEnvironment; `sre_agent_sandbox.simulated_system` → BASELINE_METRICS, SERVICE_NAMES |
| `test_server.py` | `sre_agent_sandbox.server.app` → app |
| `test_renderer.py` | `sre_agent_sandbox.models` → SREAction; `sre_agent_sandbox.renderer` → render; `sre_agent_sandbox.server.environment` → SREEnvironment; `sre_agent_sandbox.simulated_system` → SERVICE_NAMES |
| `test_demo.py` | `sre_agent_sandbox.demo.run_demo` → HeuristicAgent, RandomAgent, run_episode; `sre_agent_sandbox.models` → SREAction, SREObservation, SREState; `sre_agent_sandbox.server.environment` → SREEnvironment |
| `test_tasks.py` | `sre_agent_sandbox.demo.run_demo` → HeuristicAgent, RandomAgent; `sre_agent_sandbox.tasks` → TASKS, TaskConfig, _configure_env_for_task, evaluate_agent, grade, make_env |

---

## 2. Proposed New File Layout (Target State)

The repo root IS the package. No nested `sre_agent_sandbox/` directory.

```
sre-agent-sandbox/              <-- repo root (git root)
├── __init__.py                 <-- exports SREAction, SREObservation, SREState (from .models)
├── models.py                   <-- REAL definitions (from inner sre_agent_sandbox/models.py)
├── client.py                   <-- SREEnv(EnvClient) (from inner — currently root is already real-ish)
├── simulated_system.py         <-- MOVED from sre_agent_sandbox/simulated_system.py
├── chaos_engine.py             <-- MOVED from sre_agent_sandbox/chaos_engine.py
├── reward.py                   <-- MOVED from sre_agent_sandbox/reward.py
├── renderer.py                 <-- MOVED from sre_agent_sandbox/renderer.py
├── tasks.py                    <-- MOVED from sre_agent_sandbox/tasks.py
├── baseline_eval.py            <-- MOVED from sre_agent_sandbox/baseline_eval.py
├── baseline_inference.py       <-- MOVED from sre_agent_sandbox/baseline_inference.py
├── server/
│   ├── __init__.py             <-- docstring
│   ├── app.py                  <-- create_app() + main() (REAL, from inner sre_agent_sandbox/server/app.py)
│   └── environment.py          <-- SREEnvironment (MOVED from sre_agent_sandbox/server/environment.py)
├── demo/
│   ├── __init__.py             <-- docstring
│   └── run_demo.py             <-- MOVED from sre_agent_sandbox/demo/run_demo.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_simulated_system.py
│   ├── test_chaos_engine.py
│   ├── test_reward.py
│   ├── test_environment.py
│   ├── test_server.py
│   ├── test_renderer.py
│   ├── test_demo.py
│   └── test_tasks.py
├── docs/
│   ├── reward-system.md
│   └── training-guide.md
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── .gitignore
├── .dockerignore
├── README.md
└── uv.lock
```

**Key:** The `sre_agent_sandbox/` directory is completely eliminated. All its contents move to root or `server/`.

---

## 3. File-by-File Migration Plan

### 3.1 Files to MOVE (source → destination)

| # | Source (inner package) | Destination (root) | Action |
|---|---|---|---|
| 1 | `sre_agent_sandbox/models.py` | `models.py` (overwrite shim) | Replace root shim with real content |
| 2 | `sre_agent_sandbox/simulated_system.py` | `simulated_system.py` | Move to root |
| 3 | `sre_agent_sandbox/chaos_engine.py` | `chaos_engine.py` | Move to root |
| 4 | `sre_agent_sandbox/reward.py` | `reward.py` | Move to root |
| 5 | `sre_agent_sandbox/renderer.py` | `renderer.py` | Move to root |
| 6 | `sre_agent_sandbox/tasks.py` | `tasks.py` | Move to root |
| 7 | `sre_agent_sandbox/baseline_eval.py` | `baseline_eval.py` | Move to root |
| 8 | `sre_agent_sandbox/baseline_inference.py` | `baseline_inference.py` | Move to root |
| 9 | `sre_agent_sandbox/server/environment.py` | `server/environment.py` | Move to root server/ |
| 10 | `sre_agent_sandbox/server/app.py` | `server/app.py` (overwrite shim) | Replace root shim with real content |
| 11 | `sre_agent_sandbox/server/__init__.py` | `server/__init__.py` (overwrite) | Replace with inner's docstring |
| 12 | `sre_agent_sandbox/demo/run_demo.py` | `demo/run_demo.py` (overwrite shim) | Replace root shim with real content |
| 13 | `sre_agent_sandbox/demo/__init__.py` | `demo/__init__.py` (overwrite) | Replace with inner's docstring |

### 3.2 Files/Dirs to DELETE after migration

| Path | Reason |
|---|---|
| `sre_agent_sandbox/` (entire directory) | Inner package eliminated; all code moved to root |
| `sre_agent_sandbox/server/Dockerfile` | Duplicate of root Dockerfile |
| `sre_agent_sandbox/__init__.py` | Empty docstring, no longer needed |
| `sre_agent_sandbox/__pycache__/` | Build artifacts |

### 3.3 Files to UPDATE (root `__init__.py`)

**`__init__.py`** (root) — Change from shim to real exports:
```python
# BEFORE:
from sre_agent_sandbox.models import SREAction, SREObservation, SREState

# AFTER:
from models import SREAction, SREObservation, SREState
```

**`client.py`** (root, already has real code) — Fix imports:
```python
# BEFORE:
from sre_agent_sandbox.models import SREAction, SREObservation, SREState

# AFTER:
from models import SREAction, SREObservation, SREState
```

### 3.4 Import Changes Per File (all `sre_agent_sandbox.X` → `X`)

Every import that currently says `from sre_agent_sandbox.X import Y` must change to `from X import Y` (since root IS the package and `pythonpath = ["."]` in pyproject.toml).

#### Root-level modules:

| File | Old Import | New Import |
|---|---|---|
| `__init__.py` | `from sre_agent_sandbox.models import ...` | `from models import SREAction, SREObservation, SREState` |
| `client.py` | `from sre_agent_sandbox.models import ...` | `from models import SREAction, SREObservation, SREState` |
| `chaos_engine.py` | `from sre_agent_sandbox.simulated_system import ...` | `from simulated_system import DEPENDENCIES, SERVICE_NAMES, SimulatedSystem` |
| `reward.py` | `from sre_agent_sandbox.models import SREAction` | `from models import SREAction` |
| `reward.py` | `from sre_agent_sandbox.simulated_system import ...` | `from simulated_system import SERVICE_NAMES, SimulatedSystem` |
| `renderer.py` | `from sre_agent_sandbox.simulated_system import SERVICE_NAMES` | `from simulated_system import SERVICE_NAMES` |
| `renderer.py` | `from sre_agent_sandbox.server.environment import SREEnvironment` (TYPE_CHECKING) | `from server.environment import SREEnvironment` |
| `tasks.py` | `from sre_agent_sandbox.server.environment import SREEnvironment` | `from server.environment import SREEnvironment` |
| `baseline_eval.py` | `from sre_agent_sandbox.demo.run_demo import ...` | `from demo.run_demo import HeuristicAgent, RandomAgent` |
| `baseline_eval.py` | `from sre_agent_sandbox.tasks import ...` | `from tasks import TASKS, evaluate_agent` |
| `baseline_inference.py` | `from sre_agent_sandbox.models import ...` | `from models import SREAction, SREObservation, SREState` |
| `baseline_inference.py` | `from sre_agent_sandbox.tasks import ...` | `from tasks import TASKS, _configure_env_for_task, grade, make_env` |

#### server/ subpackage:

| File | Old Import | New Import |
|---|---|---|
| `server/app.py` | `from sre_agent_sandbox.models import SREAction, SREObservation` | `from models import SREAction, SREObservation` |
| `server/app.py` | `from sre_agent_sandbox.server.environment import SREEnvironment` | `from server.environment import SREEnvironment` |
| `server/environment.py` | `from sre_agent_sandbox.chaos_engine import ChaosEngine` | `from chaos_engine import ChaosEngine` |
| `server/environment.py` | `from sre_agent_sandbox.models import SREAction, SREObservation, SREState` | `from models import SREAction, SREObservation, SREState` |
| `server/environment.py` | `from sre_agent_sandbox.reward import RewardCalculator` | `from reward import RewardCalculator` |
| `server/environment.py` | `from sre_agent_sandbox.simulated_system import SERVICE_NAMES, SimulatedSystem` | `from simulated_system import SERVICE_NAMES, SimulatedSystem` |

#### demo/ subpackage:

| File | Old Import | New Import |
|---|---|---|
| `demo/run_demo.py` | `from sre_agent_sandbox.models import SREAction, SREObservation, SREState` | `from models import SREAction, SREObservation, SREState` |
| `demo/run_demo.py` | `from sre_agent_sandbox.server.environment import SREEnvironment` | `from server.environment import SREEnvironment` |
| `demo/run_demo.py` | `from sre_agent_sandbox.simulated_system import SERVICE_NAMES` | `from simulated_system import SERVICE_NAMES` |
| `demo/run_demo.py` (lazy) | `from sre_agent_sandbox.renderer import render as render_fn` | `from renderer import render as render_fn` |

#### tests/:

| File | Old Import | New Import |
|---|---|---|
| `test_models.py` | `from sre_agent_sandbox.models import SREAction, SREObservation, SREState` | `from models import SREAction, SREObservation, SREState` |
| `test_simulated_system.py` | `from sre_agent_sandbox.simulated_system import SimulatedSystem` | `from simulated_system import SimulatedSystem` |
| `test_chaos_engine.py` | `from sre_agent_sandbox.chaos_engine import ChaosEngine` | `from chaos_engine import ChaosEngine` |
| `test_chaos_engine.py` | `from sre_agent_sandbox.simulated_system import BASELINE_METRICS, SERVICE_NAMES, SimulatedSystem` | `from simulated_system import BASELINE_METRICS, SERVICE_NAMES, SimulatedSystem` |
| `test_reward.py` | `from sre_agent_sandbox.models import SREAction` | `from models import SREAction` |
| `test_reward.py` | `from sre_agent_sandbox.reward import RewardCalculator` | `from reward import RewardCalculator` |
| `test_reward.py` | `from sre_agent_sandbox.simulated_system import SimulatedSystem` | `from simulated_system import SimulatedSystem` |
| `test_environment.py` | `from sre_agent_sandbox.models import SREAction, SREObservation, SREState` | `from models import SREAction, SREObservation, SREState` |
| `test_environment.py` | `from sre_agent_sandbox.server.environment import SREEnvironment` | `from server.environment import SREEnvironment` |
| `test_environment.py` | `from sre_agent_sandbox.simulated_system import BASELINE_METRICS, SERVICE_NAMES` | `from simulated_system import BASELINE_METRICS, SERVICE_NAMES` |
| `test_server.py` | `from sre_agent_sandbox.server.app import app` | `from server.app import app` |
| `test_renderer.py` | `from sre_agent_sandbox.models import SREAction` | `from models import SREAction` |
| `test_renderer.py` | `from sre_agent_sandbox.renderer import render` | `from renderer import render` |
| `test_renderer.py` | `from sre_agent_sandbox.server.environment import SREEnvironment` | `from server.environment import SREEnvironment` |
| `test_renderer.py` | `from sre_agent_sandbox.simulated_system import SERVICE_NAMES` | `from simulated_system import SERVICE_NAMES` |
| `test_demo.py` | `from sre_agent_sandbox.demo.run_demo import HeuristicAgent, RandomAgent, run_episode` | `from demo.run_demo import HeuristicAgent, RandomAgent, run_episode` |
| `test_demo.py` | `from sre_agent_sandbox.models import SREAction, SREObservation, SREState` | `from models import SREAction, SREObservation, SREState` |
| `test_demo.py` | `from sre_agent_sandbox.server.environment import SREEnvironment` | `from server.environment import SREEnvironment` |
| `test_tasks.py` | `from sre_agent_sandbox.demo.run_demo import HeuristicAgent, RandomAgent` | `from demo.run_demo import HeuristicAgent, RandomAgent` |
| `test_tasks.py` | `from sre_agent_sandbox.tasks import ...` | `from tasks import TASKS, TaskConfig, _configure_env_for_task, evaluate_agent, grade, make_env` |

### 3.5 Configuration File Changes

#### `pyproject.toml`

```toml
# BEFORE:
[project.scripts]
sre-agent-demo = "sre_agent_sandbox.demo.run_demo:main"
server = "server.app:main"

# AFTER:
[project.scripts]
sre-agent-demo = "demo.run_demo:main"
server = "server.app:main"
```

The `server` script is already correct. Only `sre-agent-demo` needs updating.

The `pythonpath = ["."]` setting is already correct and critical for this flat structure.

#### `openenv.yaml`

```yaml
# BEFORE:
entry_point: sre_agent_sandbox.server.environment:SREEnvironment
grader: sre_agent_sandbox.tasks:grade

# AFTER:
entry_point: server.environment:SREEnvironment
grader: tasks:grade
```

All three task entries have `grader: sre_agent_sandbox.tasks:grade` → change to `grader: tasks:grade`.

#### `Dockerfile`

```dockerfile
# BEFORE:
COPY sre_agent_sandbox/ sre_agent_sandbox/
# ...
COPY --from=builder /app/sre_agent_sandbox /app/sre_agent_sandbox
# ...
CMD ["uvicorn", "sre_agent_sandbox.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

# AFTER:
# Copy all source files (flat package at root)
COPY models.py client.py simulated_system.py chaos_engine.py reward.py renderer.py tasks.py baseline_eval.py baseline_inference.py __init__.py ./
COPY server/ server/
COPY demo/ demo/
# ...
COPY --from=builder /app/models.py /app/client.py /app/simulated_system.py /app/chaos_engine.py /app/reward.py /app/renderer.py /app/tasks.py /app/baseline_eval.py /app/baseline_inference.py /app/__init__.py /app/
COPY --from=builder /app/server /app/server
COPY --from=builder /app/demo /app/demo
# ...
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Delete `sre_agent_sandbox/server/Dockerfile` (duplicate).

---

## 4. Risk Assessment

### 4.1 `openenv validate` Requirements

**HIGH RISK**: The `openenv.yaml` `entry_point` field is the primary way OpenEnv discovers the environment class. Changing from `sre_agent_sandbox.server.environment:SREEnvironment` to `server.environment:SREEnvironment` MUST work with `openenv validate`. This depends on:
- Whether `openenv validate` adds the CWD to `sys.path` (likely, since it's standard Python).
- Whether `openenv push` expects a specific package structure.

**Mitigation**: After restructuring, run `openenv validate` immediately to confirm. If it fails, may need to keep `sre_agent_sandbox` as the importable package name via `pyproject.toml` package config (e.g., using hatchling's `[tool.hatch.build.targets.wheel]` with `packages = ["."]` or similar).

### 4.2 Package Discovery by Hatchling

**MEDIUM RISK**: The `[build-system]` uses hatchling. Currently hatchling auto-discovers `sre_agent_sandbox/` as the package (because it has `__init__.py`). After restructuring, with `__init__.py` at root, hatchling may need explicit configuration:

```toml
[tool.hatch.build.targets.wheel]
packages = ["."]
```

Or the project may need to use `find` mode:

```toml
[tool.hatch.build.targets.wheel]
packages = ["models", "client", "server", "demo", ...]
```

This needs testing. The simplest approach: since this is not distributed as a pip package (it's an OpenEnv environment run via Docker/uvicorn), package build config may not matter in practice.

### 4.3 Circular Import Issues

**LOW RISK**: No circular imports exist in the current codebase. The dependency graph is a clean DAG:
```
models.py  ← (no internal deps, only openenv/pydantic)
simulated_system.py  ← (no internal deps, only stdlib)
chaos_engine.py  ← simulated_system
reward.py  ← models, simulated_system
renderer.py  ← simulated_system, server.environment (TYPE_CHECKING only)
server/environment.py  ← chaos_engine, models, reward, simulated_system
tasks.py  ← server.environment, renderer (lazy import)
demo/run_demo.py  ← models, server.environment, simulated_system, renderer (lazy)
baseline_eval.py  ← demo.run_demo, tasks
baseline_inference.py  ← models, tasks
server/app.py  ← models, server.environment
```

The `renderer.py` → `server.environment` import is behind `TYPE_CHECKING`, so it's safe. The lazy imports of `renderer` in `tasks.py` and `demo/run_demo.py` also avoid cycles.

### 4.4 Test Discovery

**LOW RISK**: `pyproject.toml` already has:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

The `pythonpath = ["."]` ensures all root-level modules are importable. Tests will work with the new flat import paths.

### 4.5 Docker Build

**MEDIUM RISK**: The Dockerfile needs significant changes to COPY individual files instead of a single directory. This is more fragile (new files must be added to COPY). Consider using a `.dockerignore`-aware approach:

```dockerfile
COPY . .
```

...with appropriate `.dockerignore` to exclude `.venv`, `.git`, `tests/`, etc. This is simpler and more maintainable.

### 4.6 Script Entry Points

**LOW RISK**: `server = "server.app:main"` already works (no `sre_agent_sandbox.` prefix). Only `sre-agent-demo` needs updating from `sre_agent_sandbox.demo.run_demo:main` to `demo.run_demo:main`.

### 4.7 Module Invocation (`python -m`)

**LOW RISK**: Several files have `if __name__ == "__main__"` blocks. These will work as `python -m baseline_eval`, `python -m baseline_inference`, etc., instead of `python -m sre_agent_sandbox.baseline_eval`.

---

## 5. Execution Order (Recommended)

1. **Move inner files to root** (overwriting shims where they exist)
2. **Add new root files** (simulated_system.py, chaos_engine.py, etc.)
3. **Move server/environment.py** to root server/
4. **Move demo/run_demo.py** to root demo/
5. **Update ALL imports** (bulk find-and-replace: `sre_agent_sandbox.` → empty)
6. **Update pyproject.toml** (script entry point)
7. **Update openenv.yaml** (entry_point, grader paths)
8. **Update Dockerfile** (COPY paths, CMD)
9. **Delete `sre_agent_sandbox/`** directory entirely
10. **Run tests**: `uv run pytest tests/ -v`
11. **Run linter**: `uv run ruff check .`
12. **Run openenv validate** (if available)

---

## 6. Summary of Change Counts

| Category | Count |
|---|---|
| Files to move/overwrite | 13 |
| Files to delete (inner package) | Entire `sre_agent_sandbox/` dir (~15 files + __pycache__) |
| Import statements to change | ~40 across ~20 files |
| Config files to update | 3 (pyproject.toml, openenv.yaml, Dockerfile) |
| New files to create | 0 |
