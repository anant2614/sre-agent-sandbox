"""SRE Agent Sandbox Environment Client.

Typed OpenEnv client for interacting with a remote SRE environment server.

Usage::

    from client import SREEnv
    from models import SREAction

    async with SREEnv(base_url="http://localhost:8000") as client:
        result = await client.reset(seed=42)
        result = await client.step(SREAction(action_type=1, target_service="db"))
        print(result.observation.health_status)
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import SREAction, SREObservation, SREState


class SREEnv(EnvClient[SREAction, SREObservation, State]):
    """Client for the SRE Agent Sandbox environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example::

        async with SREEnv(base_url="http://localhost:8000") as client:
            result = await client.reset(seed=42)
            result = await client.step(SREAction(action_type=1, target_service="db"))
    """

    def _step_payload(self, action: SREAction) -> Dict:
        return {
            "action_type": action.action_type,
            "target_service": action.target_service,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SREObservation]:
        obs_data = payload.get("observation", {})
        observation = SREObservation(
            metrics=obs_data.get("metrics", {}),
            log_buffer=obs_data.get("log_buffer", []),
            health_status=obs_data.get("health_status", {}),
            active_alerts=obs_data.get("active_alerts", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
