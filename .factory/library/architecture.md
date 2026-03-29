# Architecture

Architectural decisions and patterns for the SRE-Agent-Sandbox.

---

## Project Structure (OpenEnv Convention)
```
sre_agent_sandbox/
├── __init__.py           # Exports Action, Observation, EnvClient
├── models.py             # SREAction, SREObservation, SREState
├── chaos_engine.py       # ChaosEngine class
├── simulated_system.py   # 3-tier system state model
├── reward.py             # Reward computation (5 components)
├── client.py             # EnvClient subclass (optional)
├── openenv.yaml          # Environment manifest
├── pyproject.toml
├── server/
│   ├── environment.py    # SREEnvironment(Environment) — reset/step/state
│   ├── app.py            # FastAPI app with REST + WebSocket
│   └── Dockerfile
├── demo/
│   └── run_demo.py       # Random + heuristic agent demo
└── tests/
    ├── test_models.py
    ├── test_chaos_engine.py
    ├── test_reward.py
    ├── test_environment.py
    └── test_server.py
```

## Key Design Decisions
- OpenEnv base classes: Action(extra="forbid"), Observation(extra="forbid"), State(extra="allow")
- All simulation in-memory (no real I/O)
- ChaosEngine uses seeded random.Random for deterministic replay
- Reward computed as sum of 5 independent components
- 3-tier system: API Gateway -> Order Service -> DB/Cache (dependency chain)
- Episode termination: max_steps=200 or total meltdown (all services down)
