"""Tests for SRE Agent Sandbox Pydantic models.

Tests cover:
- Valid construction for all 15 action combinations (5 action_types x 3 services)
- Invalid action_type rejection (5, -1, string)
- Invalid target_service rejection
- Missing required fields
- Extra fields rejected (extra='forbid')
- Serialization round-trip (model_dump_json -> model_validate_json)
- SREObservation field validation
- SREState field validation
- Inheritance from OpenEnv base classes
"""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from sre_agent_sandbox.models import SREAction, SREObservation, SREState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = [0, 1, 2, 3, 4]
VALID_TARGET_SERVICES = ["api", "order", "db"]

SAMPLE_METRICS: dict[str, dict[str, float]] = {
    "api": {"cpu": 45.0, "memory": 60.0, "latency": 50.0, "request_count": 1000.0},
    "order": {"cpu": 30.0, "memory": 40.0, "latency": 80.0, "request_count": 500.0},
    "db": {"cpu": 55.0, "memory": 70.0, "latency": 20.0, "request_count": 2000.0},
}

SAMPLE_HEALTH_STATUS: dict[str, bool] = {"api": True, "order": True, "db": True}


# ---------------------------------------------------------------------------
# SREAction Tests
# ---------------------------------------------------------------------------


class TestSREActionValidConstruction:
    """Test all 15 valid action combinations construct without error."""

    @pytest.mark.parametrize(
        "action_type,target_service",
        list(itertools.product(VALID_ACTION_TYPES, VALID_TARGET_SERVICES)),
    )
    def test_valid_action_combos(self, action_type: int, target_service: str) -> None:
        action = SREAction(action_type=action_type, target_service=target_service)
        assert action.action_type == action_type
        assert action.target_service == target_service

    def test_action_inherits_from_openenv_action(self) -> None:
        from openenv.core.env_server.types import Action

        action = SREAction(action_type=0, target_service="api")
        assert isinstance(action, Action)

    def test_action_has_metadata_from_base(self) -> None:
        action = SREAction(action_type=0, target_service="api")
        assert action.metadata == {}


class TestSREActionInvalidActionType:
    """Test that invalid action_type values are rejected."""

    def test_action_type_5_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=5, target_service="api")

    def test_action_type_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=-1, target_service="api")

    def test_action_type_6_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=6, target_service="api")

    def test_action_type_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type="restart", target_service="api")  # type: ignore[arg-type]

    def test_action_type_float_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=1.5, target_service="api")  # type: ignore[arg-type]

    def test_action_type_none_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=None, target_service="api")  # type: ignore[arg-type]


class TestSREActionInvalidTargetService:
    """Test that invalid target_service values are rejected."""

    def test_unknown_service_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0, target_service="unknown")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0, target_service="")

    def test_numeric_service_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0, target_service=123)  # type: ignore[arg-type]

    def test_similar_but_wrong_service_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0, target_service="API")  # case-sensitive

    def test_database_not_valid_service(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0, target_service="database")


class TestSREActionMissingFields:
    """Test that missing required fields are rejected."""

    def test_missing_action_type(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(target_service="api")  # type: ignore[call-arg]

    def test_missing_target_service(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0)  # type: ignore[call-arg]

    def test_missing_both_fields(self) -> None:
        with pytest.raises(ValidationError):
            SREAction()  # type: ignore[call-arg]


class TestSREActionExtraFields:
    """Test that extra fields are rejected (extra='forbid')."""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0, target_service="api", bogus_field="x")

    def test_extra_field_priority_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0, target_service="api", priority=1)

    def test_extra_field_description_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREAction(action_type=0, target_service="api", description="test")


class TestSREActionSerialization:
    """Test JSON serialization round-trip for SREAction."""

    @pytest.mark.parametrize(
        "action_type,target_service",
        [(0, "api"), (2, "order"), (4, "db")],
    )
    def test_serialization_roundtrip(self, action_type: int, target_service: str) -> None:
        original = SREAction(action_type=action_type, target_service=target_service)
        json_str = original.model_dump_json()
        restored = SREAction.model_validate_json(json_str)
        assert restored.action_type == original.action_type
        assert restored.target_service == original.target_service
        assert restored == original

    def test_model_dump_contains_expected_keys(self) -> None:
        action = SREAction(action_type=1, target_service="api")
        dumped = action.model_dump()
        assert "action_type" in dumped
        assert "target_service" in dumped
        assert "metadata" in dumped


# ---------------------------------------------------------------------------
# SREObservation Tests
# ---------------------------------------------------------------------------


class TestSREObservationValidConstruction:
    """Test valid SREObservation construction."""

    def test_basic_construction(self) -> None:
        obs = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=["System started"],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=[],
        )
        assert obs.metrics == SAMPLE_METRICS
        assert obs.log_buffer == ["System started"]
        assert obs.health_status == SAMPLE_HEALTH_STATUS
        assert obs.active_alerts == []

    def test_inherits_done_from_observation(self) -> None:
        obs = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=[],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=[],
            done=True,
            reward=5.0,
        )
        assert obs.done is True
        assert obs.reward == 5.0

    def test_inherits_from_openenv_observation(self) -> None:
        from openenv.core.env_server.types import Observation

        obs = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=[],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=[],
        )
        assert isinstance(obs, Observation)

    def test_default_done_is_false(self) -> None:
        obs = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=[],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=[],
        )
        assert obs.done is False

    def test_default_reward_is_none(self) -> None:
        obs = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=[],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=[],
        )
        assert obs.reward is None

    def test_log_buffer_max_10(self) -> None:
        """log_buffer should accept up to 10 entries."""
        obs = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=[f"log {i}" for i in range(10)],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=[],
        )
        assert len(obs.log_buffer) == 10

    def test_log_buffer_over_10_rejected(self) -> None:
        """log_buffer with more than 10 entries should be rejected."""
        with pytest.raises(ValidationError):
            SREObservation(
                metrics=SAMPLE_METRICS,
                log_buffer=[f"log {i}" for i in range(11)],
                health_status=SAMPLE_HEALTH_STATUS,
                active_alerts=[],
            )

    def test_has_metadata_from_base(self) -> None:
        obs = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=[],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=[],
        )
        assert obs.metadata == {}

    def test_with_active_alerts(self) -> None:
        obs = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=[],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=["High CPU on api", "Memory leak on db"],
        )
        assert len(obs.active_alerts) == 2


class TestSREObservationMissingFields:
    """Test that missing required fields are rejected."""

    def test_missing_metrics(self) -> None:
        with pytest.raises(ValidationError):
            SREObservation(
                log_buffer=[],
                health_status=SAMPLE_HEALTH_STATUS,
                active_alerts=[],
            )  # type: ignore[call-arg]

    def test_missing_log_buffer(self) -> None:
        with pytest.raises(ValidationError):
            SREObservation(
                metrics=SAMPLE_METRICS,
                health_status=SAMPLE_HEALTH_STATUS,
                active_alerts=[],
            )  # type: ignore[call-arg]

    def test_missing_health_status(self) -> None:
        with pytest.raises(ValidationError):
            SREObservation(
                metrics=SAMPLE_METRICS,
                log_buffer=[],
                active_alerts=[],
            )  # type: ignore[call-arg]

    def test_missing_active_alerts(self) -> None:
        with pytest.raises(ValidationError):
            SREObservation(
                metrics=SAMPLE_METRICS,
                log_buffer=[],
                health_status=SAMPLE_HEALTH_STATUS,
            )  # type: ignore[call-arg]


class TestSREObservationExtraFields:
    """Test that extra fields are rejected (extra='forbid')."""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREObservation(
                metrics=SAMPLE_METRICS,
                log_buffer=[],
                health_status=SAMPLE_HEALTH_STATUS,
                active_alerts=[],
                unknown_field="x",
            )

    def test_extra_field_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREObservation(
                metrics=SAMPLE_METRICS,
                log_buffer=[],
                health_status=SAMPLE_HEALTH_STATUS,
                active_alerts=[],
                timestamp=123456,
            )


class TestSREObservationSerialization:
    """Test JSON serialization round-trip for SREObservation."""

    def test_serialization_roundtrip(self) -> None:
        original = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=["event1", "event2"],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=["High CPU on api"],
            done=False,
            reward=3.5,
        )
        json_str = original.model_dump_json()
        restored = SREObservation.model_validate_json(json_str)
        assert restored == original

    def test_serialization_roundtrip_empty_lists(self) -> None:
        original = SREObservation(
            metrics=SAMPLE_METRICS,
            log_buffer=[],
            health_status=SAMPLE_HEALTH_STATUS,
            active_alerts=[],
        )
        json_str = original.model_dump_json()
        restored = SREObservation.model_validate_json(json_str)
        assert restored == original


# ---------------------------------------------------------------------------
# SREState Tests
# ---------------------------------------------------------------------------


class TestSREStateValidConstruction:
    """Test valid SREState construction."""

    def test_basic_construction(self) -> None:
        state = SREState(
            system_health_score=0.95,
            active_incidents=[],
        )
        assert state.system_health_score == 0.95
        assert state.active_incidents == []

    def test_inherits_episode_id_from_state(self) -> None:
        state = SREState(
            episode_id="ep-001",
            step_count=5,
            system_health_score=0.8,
            active_incidents=["memory_leak_db"],
        )
        assert state.episode_id == "ep-001"
        assert state.step_count == 5

    def test_inherits_from_openenv_state(self) -> None:
        from openenv.core.env_server.types import State

        state = SREState(
            system_health_score=1.0,
            active_incidents=[],
        )
        assert isinstance(state, State)

    def test_default_episode_id_is_none(self) -> None:
        state = SREState(
            system_health_score=1.0,
            active_incidents=[],
        )
        assert state.episode_id is None

    def test_default_step_count_is_zero(self) -> None:
        state = SREState(
            system_health_score=1.0,
            active_incidents=[],
        )
        assert state.step_count == 0

    def test_system_health_score_boundary_0(self) -> None:
        state = SREState(system_health_score=0.0, active_incidents=[])
        assert state.system_health_score == 0.0

    def test_system_health_score_boundary_1(self) -> None:
        state = SREState(system_health_score=1.0, active_incidents=[])
        assert state.system_health_score == 1.0

    def test_system_health_score_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREState(system_health_score=-0.1, active_incidents=[])

    def test_system_health_score_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SREState(system_health_score=1.1, active_incidents=[])

    def test_with_active_incidents(self) -> None:
        state = SREState(
            system_health_score=0.5,
            active_incidents=["memory_leak_db", "bad_config_api"],
        )
        assert len(state.active_incidents) == 2


class TestSREStateMissingFields:
    """Test that missing required fields are rejected."""

    def test_missing_system_health_score(self) -> None:
        with pytest.raises(ValidationError):
            SREState(active_incidents=[])  # type: ignore[call-arg]

    def test_missing_active_incidents(self) -> None:
        with pytest.raises(ValidationError):
            SREState(system_health_score=1.0)  # type: ignore[call-arg]


class TestSREStateExtraFields:
    """Test that State uses extra='allow' per OpenEnv convention."""

    def test_extra_fields_allowed_on_state(self) -> None:
        """State uses extra='allow' per OpenEnv convention, so extra fields should be accepted."""
        state = SREState(
            system_health_score=1.0,
            active_incidents=[],
            custom_data="allowed",
        )
        assert state.system_health_score == 1.0


class TestSREStateSerialization:
    """Test JSON serialization round-trip for SREState."""

    def test_serialization_roundtrip(self) -> None:
        original = SREState(
            episode_id="ep-042",
            step_count=10,
            system_health_score=0.75,
            active_incidents=["latent_dep_order"],
        )
        json_str = original.model_dump_json()
        restored = SREState.model_validate_json(json_str)
        assert restored.episode_id == original.episode_id
        assert restored.step_count == original.step_count
        assert restored.system_health_score == original.system_health_score
        assert restored.active_incidents == original.active_incidents

    def test_serialization_roundtrip_defaults(self) -> None:
        original = SREState(
            system_health_score=1.0,
            active_incidents=[],
        )
        json_str = original.model_dump_json()
        restored = SREState.model_validate_json(json_str)
        assert restored.system_health_score == original.system_health_score
        assert restored.active_incidents == original.active_incidents
