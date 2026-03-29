"""Pydantic models for the SRE Agent Sandbox environment.

Defines SREAction, SREObservation, and SREState as subclasses of the
OpenEnv base types (Action, Observation, State).
"""

from __future__ import annotations

from typing import Dict, List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator

# Required service keys for metrics and health_status
VALID_SERVICE_KEYS = frozenset({"api", "order", "db"})

# Required metric sub-keys per service
VALID_METRIC_KEYS = frozenset({"cpu", "memory", "latency", "request_count"})


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

    @field_validator("action_type", mode="before")
    @classmethod
    def _strict_int_action_type(cls, v: object) -> object:
        """Reject float-to-int coercion (e.g., 1.0) and bool values."""
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"action_type must be a strict int, got {type(v).__name__}"
            )
        return v


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

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, v: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Enforce that metrics has exactly keys {api, order, db}, each with {cpu, memory, latency, request_count}."""
        if set(v.keys()) != VALID_SERVICE_KEYS:
            raise ValueError(
                f"metrics must have exactly keys {sorted(VALID_SERVICE_KEYS)}, "
                f"got {sorted(v.keys())}"
            )
        for svc, svc_metrics in v.items():
            if set(svc_metrics.keys()) != VALID_METRIC_KEYS:
                raise ValueError(
                    f"metrics['{svc}'] must have exactly keys {sorted(VALID_METRIC_KEYS)}, "
                    f"got {sorted(svc_metrics.keys())}"
                )
        return v

    @field_validator("health_status")
    @classmethod
    def _validate_health_status(cls, v: Dict[str, bool]) -> Dict[str, bool]:
        """Enforce that health_status has exactly keys {api, order, db} with bool values."""
        if set(v.keys()) != VALID_SERVICE_KEYS:
            raise ValueError(
                f"health_status must have exactly keys {sorted(VALID_SERVICE_KEYS)}, "
                f"got {sorted(v.keys())}"
            )
        return v


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
