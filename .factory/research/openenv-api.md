# OpenEnv API Reference

## Package: openenv-core (PyPI), v0.2.2+
GitHub: https://github.com/meta-pytorch/OpenEnv

## Base Classes (from openenv.core.env_server.types)

### Action
```python
from pydantic import BaseModel, ConfigDict, Field

class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Observation
```python
class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    done: bool = Field(default=False)
    reward: bool | int | float | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### State
```python
class State(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)
    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)
```

## Environment (from openenv.core.env_server.interfaces)
```python
class Environment(ABC, Generic[ActT, ObsT, StateT]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    @abstractmethod
    def reset(self, seed=None, episode_id=None, **kwargs) -> ObsT: ...

    @abstractmethod
    def step(self, action: ActT, timeout_s=None, **kwargs) -> ObsT: ...

    @property
    @abstractmethod
    def state(self) -> StateT: ...

    def get_metadata(self) -> EnvironmentMetadata: ...
    def close(self) -> None: ...
```

## WebSocket Protocol
- Client -> Server: {"type": "reset"/"step"/"state"/"close", "data": {...}}
- Server -> Client: {"type": "observation"/"state"/"error", "data": {...}}

## HTTP Endpoints
- POST /reset, POST /step, GET /state, GET /health, GET /schema, WS /ws

## Server Creation (app.py)
```python
from openenv.core.env_server.server import create_app
from your_env import YourEnvironment
app = create_app(YourEnvironment)
```
Or manual FastAPI setup if create_app isn't available.

## Standard Project Structure
```
my_env/
├── __init__.py
├── models.py         # Action, Observation, State subclasses
├── client.py         # EnvClient subclass
├── openenv.yaml      # Environment manifest
├── pyproject.toml
└── server/
    ├── environment.py  # Environment subclass
    ├── app.py          # FastAPI app
    └── Dockerfile
```

## Key Patterns
- Async-first client (use .sync() for synchronous)
- Pydantic v2 with extra="forbid" on Action/Observation, extra="allow" on State
- Each environment has its own pyproject.toml
- Docker: two-stage build with uv
