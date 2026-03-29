"""Pydantic models for the SRE Agent Sandbox environment.

Defines SREAction, SREObservation, and SREState as subclasses of the
OpenEnv base types (Action, Observation, State).
"""

from __future__ import annotations

from typing import Dict, List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SREAction(Action):
    """An action the SRE agent can take on a target service.

    Actions:
        0 = NoOp (do nothing)
        1 = RestartService
        2 = Rollback
        3 = ScaleUp
        4 = ClearCache

    Target services: api, order, db
    """

    action_type: Literal[0, 1, 2, 3, 4] = Field(
        ...,
        description="Action to perform: 0=NoOp, 1=Restart, 2=Rollback, 3=ScaleUp, 4=ClearCache",
    )
    target_service: Literal["api", "order", "db"] = Field(
        ...,
        description="Target service for the action",
    )


class SREObservation(Observation):
    """Observation returned by the SRE environment after each step.

    Inherits ``done`` (bool) and ``reward`` (float | None) from the
    OpenEnv ``Observation`` base class.
    """

    metrics: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Per-service metrics: {service: {cpu, memory, latency, request_count}}",
    )
    log_buffer: List[str] = Field(
        ...,
        max_length=10,
        description="Rolling log buffer (max 10 entries, FIFO)",
    )
    health_status: Dict[str, bool] = Field(
        ...,
        description="Per-service health: {api: bool, order: bool, db: bool}",
    )
    active_alerts: List[str] = Field(
        ...,
        description="Currently active alert messages",
    )


class SREState(State):
    """Internal environment state for the SRE simulation.

    Inherits ``episode_id`` (Optional[str]) and ``step_count`` (int >= 0)
    from the OpenEnv ``State`` base class.  State uses ``extra='allow'``
    per OpenEnv convention.
    """

    system_health_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall system health score (0.0 = total failure, 1.0 = fully healthy)",
    )
    active_incidents: List[str] = Field(
        ...,
        description="List of currently active incident identifiers",
    )
