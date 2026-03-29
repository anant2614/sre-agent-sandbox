"""Tests for SREEnvironment — the main environment class.

Covers:
  - reset() returns valid SREObservation with healthy baseline
  - reset() is deterministic with same seed
  - reset() clears all episode state
  - step() returns valid SREObservation and increments step_count
  - step() before reset() raises RuntimeError
  - step() after done raises RuntimeError
  - Episode terminates at max_steps (200)
  - Episode terminates on total meltdown
  - state property returns valid SREState
  - Deterministic replay with same seed and same actions
  - Performance: step() completes in <10ms average
  - NoOp has no remediation effect
  - Correct remediation resolves fault
  - Mismatched remediation does not resolve fault
  - log_buffer capped at 10
  - Actions scoped to target service only
  - SUPPORTS_CONCURRENT_SESSIONS is True
  - render() produces ASCII dashboard
"""

from __future__ import annotations

import time

import pytest

from sre_agent_sandbox.models import SREAction, SREObservation, SREState
from sre_agent_sandbox.server.environment import SREEnvironment
from sre_agent_sandbox.simulated_system import BASELINE_METRICS, SERVICE_NAMES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop(target: str = "api") -> SREAction:
    return SREAction(action_type=0, target_service=target)


def _restart(target: str) -> SREAction:
    return SREAction(action_type=1, target_service=target)


def _rollback(target: str) -> SREAction:
    return SREAction(action_type=2, target_service=target)


def _scaleup(target: str) -> SREAction:
    return SREAction(action_type=3, target_service=target)


def _clearcache(target: str) -> SREAction:
    return SREAction(action_type=4, target_service=target)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> SREEnvironment:
    """Return a fresh, un-reset environment."""
    return SREEnvironment()


@pytest.fixture
def ready_env() -> SREEnvironment:
    """Return an environment that has been reset with a fixed seed."""
    e = SREEnvironment()
    e.reset(seed=42)
    return e


# ===================================================================
# 1. reset() returns valid SREObservation with healthy baseline
# ===================================================================

class TestResetReturnsValidObservation:
    def test_reset_returns_observation_type(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        assert isinstance(obs, SREObservation)

    def test_reset_observation_done_is_false(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        assert obs.done is False

    def test_reset_observation_reward_is_numeric(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        # reward can be None or numeric; at reset it's typically None or 0
        assert obs.reward is None or isinstance(obs.reward, (int, float))

    def test_reset_metrics_has_all_services(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        for svc in SERVICE_NAMES:
            assert svc in obs.metrics

    def test_reset_metrics_keys_per_service(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        for svc in SERVICE_NAMES:
            for key in ("cpu", "memory", "latency", "request_count"):
                assert key in obs.metrics[svc]

    def test_reset_metrics_in_physical_bounds(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        for svc in SERVICE_NAMES:
            m = obs.metrics[svc]
            assert 0.0 <= m["cpu"] <= 100.0
            assert 0.0 <= m["memory"] <= 100.0
            assert m["latency"] >= 0.0
            assert m["request_count"] >= 0.0

    def test_reset_health_all_healthy(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        for svc in SERVICE_NAMES:
            assert obs.health_status[svc] is True

    def test_reset_active_alerts_empty(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        assert obs.active_alerts == []

    def test_reset_log_buffer_is_list(self, env: SREEnvironment) -> None:
        obs = env.reset(seed=42)
        assert isinstance(obs.log_buffer, list)
        assert len(obs.log_buffer) <= 10


# ===================================================================
# 2. reset() is deterministic with same seed
# ===================================================================

class TestResetDeterministic:
    def test_same_seed_same_observation(self) -> None:
        env1 = SREEnvironment()
        env2 = SREEnvironment()
        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)
        assert obs1.metrics == obs2.metrics
        assert obs1.health_status == obs2.health_status
        assert obs1.log_buffer == obs2.log_buffer
        assert obs1.active_alerts == obs2.active_alerts

    def test_different_seeds_may_differ(self) -> None:
        """Different seeds should produce at least one field difference
        after some steps (metrics drift is seeded)."""
        env1 = SREEnvironment()
        env2 = SREEnvironment()
        env1.reset(seed=42)
        env2.reset(seed=9999)
        # Take a few steps to let drift diverge
        action = _noop()
        obs1 = obs2 = None
        for _ in range(10):
            obs1 = env1.step(action)
            obs2 = env2.step(action)
        assert obs1 is not None and obs2 is not None
        # At least some metric should differ after 10 steps with different seeds
        metrics1 = obs1.metrics
        metrics2 = obs2.metrics
        any_diff = any(
            metrics1[svc][k] != metrics2[svc][k]
            for svc in SERVICE_NAMES
            for k in ("cpu", "memory", "latency", "request_count")
        )
        assert any_diff, "Different seeds should produce different metrics after steps"


# ===================================================================
# 3. reset() clears all episode state
# ===================================================================

class TestResetClearsState:
    def test_step_count_zeroed(self, env: SREEnvironment) -> None:
        env.reset(seed=42)
        for _ in range(5):
            env.step(_noop())
        assert env.state.step_count == 5
        env.reset(seed=42)
        assert env.state.step_count == 0

    def test_new_episode_id(self, env: SREEnvironment) -> None:
        env.reset(seed=42)
        old_id = env.state.episode_id
        env.reset(seed=42)
        new_id = env.state.episode_id
        # episode_id should change on reset
        assert new_id != old_id

    def test_active_incidents_cleared(self, env: SREEnvironment) -> None:
        env.reset(seed=42)
        # Run steps to potentially generate incidents
        for _ in range(10):
            env.step(_noop())
        env.reset(seed=42)
        assert env.state.active_incidents == []

    def test_health_restored_after_degradation(self, env: SREEnvironment) -> None:
        env.reset(seed=42)
        # Run some steps to let chaos potentially degrade services
        for _ in range(20):
            obs = env.step(_noop())
            if obs.done:
                break
        # Reset should restore everything
        obs = env.reset(seed=42)
        for svc in SERVICE_NAMES:
            assert obs.health_status[svc] is True


# ===================================================================
# 4. step() returns valid SREObservation and increments step_count
# ===================================================================

class TestStepBasic:
    def test_step_returns_observation(self, ready_env: SREEnvironment) -> None:
        obs = ready_env.step(_noop())
        assert isinstance(obs, SREObservation)

    def test_step_increments_step_count(self, ready_env: SREEnvironment) -> None:
        assert ready_env.state.step_count == 0
        ready_env.step(_noop())
        assert ready_env.state.step_count == 1
        ready_env.step(_noop())
        assert ready_env.state.step_count == 2

    def test_step_preserves_episode_id(self, ready_env: SREEnvironment) -> None:
        ep_id = ready_env.state.episode_id
        for _ in range(5):
            ready_env.step(_noop())
        assert ready_env.state.episode_id == ep_id

    def test_step_observation_has_reward(self, ready_env: SREEnvironment) -> None:
        obs = ready_env.step(_noop())
        assert isinstance(obs.reward, (int, float))

    def test_step_observation_has_done(self, ready_env: SREEnvironment) -> None:
        obs = ready_env.step(_noop())
        assert isinstance(obs.done, bool)

    def test_step_observation_metrics_valid(self, ready_env: SREEnvironment) -> None:
        obs = ready_env.step(_noop())
        for svc in SERVICE_NAMES:
            assert svc in obs.metrics
            for key in ("cpu", "memory", "latency", "request_count"):
                assert key in obs.metrics[svc]

    def test_ten_steps_all_valid(self, ready_env: SREEnvironment) -> None:
        for i in range(10):
            obs = ready_env.step(_noop())
            assert isinstance(obs, SREObservation)
            assert ready_env.state.step_count == i + 1


# ===================================================================
# 5. step() before reset() raises RuntimeError
# ===================================================================

class TestStepBeforeReset:
    def test_step_before_reset_raises(self, env: SREEnvironment) -> None:
        with pytest.raises(RuntimeError):
            env.step(_noop())


# ===================================================================
# 6. step() after done raises RuntimeError
# ===================================================================

class TestStepAfterDone:
    def test_step_after_max_steps_raises(self) -> None:
        env = SREEnvironment(max_steps=5)
        env.reset(seed=42)
        for _ in range(5):
            obs = env.step(_noop())
        assert obs.done is True
        with pytest.raises(RuntimeError):
            env.step(_noop())

    def test_step_after_meltdown_raises(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Force total meltdown by directly manipulating the system
        for svc in SERVICE_NAMES:
            env._system._services[svc]["is_down"] = True
            env._system._services[svc]["is_healthy"] = False
        obs = env.step(_noop())
        assert obs.done is True
        with pytest.raises(RuntimeError):
            env.step(_noop())


# ===================================================================
# 7. Episode terminates at max_steps
# ===================================================================

class TestMaxStepsTermination:
    def test_done_at_max_steps(self) -> None:
        max_s = 10
        env = SREEnvironment(max_steps=max_s)
        env.reset(seed=42)
        for i in range(max_s):
            obs = env.step(_noop())
            if i < max_s - 1:
                assert obs.done is False, f"Should not be done at step {i+1}"
        assert obs.done is True
        assert env.state.step_count == max_s

    def test_default_max_steps_is_200(self) -> None:
        # Use fault_probability=0 to avoid premature meltdown
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)
        for _ in range(200):
            obs = env.step(_noop())
        assert obs.done is True
        assert env.state.step_count == 200


# ===================================================================
# 8. Episode terminates on total meltdown (all services down)
# ===================================================================

class TestMeltdownTermination:
    def test_total_meltdown_done_true(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Force all services down
        for svc in SERVICE_NAMES:
            env._system._services[svc]["is_down"] = True
            env._system._services[svc]["is_healthy"] = False
        obs = env.step(_noop())
        assert obs.done is True

    def test_partial_meltdown_not_done(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Only one service down
        env._system._services["db"]["is_down"] = True
        env._system._services["db"]["is_healthy"] = False
        obs = env.step(_noop())
        assert obs.done is False

    def test_two_services_down_not_done(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        env._system._services["db"]["is_down"] = True
        env._system._services["db"]["is_healthy"] = False
        env._system._services["order"]["is_down"] = True
        env._system._services["order"]["is_healthy"] = False
        obs = env.step(_noop())
        assert obs.done is False


# ===================================================================
# 9. state property returns valid SREState
# ===================================================================

class TestStateProperty:
    def test_state_type(self, ready_env: SREEnvironment) -> None:
        assert isinstance(ready_env.state, SREState)

    def test_state_episode_id_non_empty(self, ready_env: SREEnvironment) -> None:
        assert ready_env.state.episode_id is not None
        assert len(ready_env.state.episode_id) > 0

    def test_state_step_count_zero_after_reset(self, ready_env: SREEnvironment) -> None:
        assert ready_env.state.step_count == 0

    def test_state_health_score_in_range(self, ready_env: SREEnvironment) -> None:
        assert 0.0 <= ready_env.state.system_health_score <= 1.0

    def test_state_health_score_after_reset_is_one(self, ready_env: SREEnvironment) -> None:
        # All healthy => health score should be 1.0
        assert ready_env.state.system_health_score == 1.0

    def test_state_active_incidents_after_reset(self, ready_env: SREEnvironment) -> None:
        assert ready_env.state.active_incidents == []

    def test_state_reflects_degradation(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Force a service down
        env._system._services["db"]["is_down"] = True
        env._system._services["db"]["is_healthy"] = False
        env._system._active_faults["db"] = "memory_leak"
        state = env.state
        assert state.system_health_score < 1.0
        assert len(state.active_incidents) > 0

    def test_state_active_incidents_reflect_faults(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        env._system._active_faults["api"] = "bad_config"
        env._system._services["api"]["is_healthy"] = False
        state = env.state
        assert any("api" in inc or "bad_config" in inc for inc in state.active_incidents)


# ===================================================================
# 10. Deterministic replay with same seed and same actions
# ===================================================================

class TestDeterministicReplay:
    def test_same_seed_same_actions_same_trajectory(self) -> None:
        actions = [
            _noop(),
            _restart("api"),
            _noop(),
            _scaleup("db"),
            _rollback("order"),
            _noop(),
            _clearcache("api"),
            _noop(),
            _noop(),
            _restart("db"),
        ]

        env1 = SREEnvironment()
        env2 = SREEnvironment()

        obs1_list = []
        obs2_list = []

        env1.reset(seed=123)
        env2.reset(seed=123)

        for action in actions:
            obs1_list.append(env1.step(action))
            obs2_list.append(env2.step(action))

        for i, (o1, o2) in enumerate(zip(obs1_list, obs2_list)):
            assert o1.metrics == o2.metrics, f"Step {i}: metrics differ"
            assert o1.health_status == o2.health_status, f"Step {i}: health_status differ"
            assert o1.done == o2.done, f"Step {i}: done differ"
            assert o1.reward == o2.reward, f"Step {i}: reward differ"
            assert o1.active_alerts == o2.active_alerts, f"Step {i}: active_alerts differ"
            assert o1.log_buffer == o2.log_buffer, f"Step {i}: log_buffer differ"


# ===================================================================
# 11. Performance: step() completes in <10ms average
# ===================================================================

class TestPerformance:
    def test_step_under_10ms_average(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        action = _noop()

        times = []
        for _ in range(100):
            start = time.perf_counter()
            obs = env.step(action)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if obs.done:
                env.reset(seed=42)

        avg_ms = (sum(times) / len(times)) * 1000
        max_ms = max(times) * 1000
        assert avg_ms < 10.0, f"Average step time {avg_ms:.2f}ms exceeds 10ms"
        assert max_ms < 50.0, f"Max step time {max_ms:.2f}ms exceeds 50ms"


# ===================================================================
# 12. NoOp has no remediation effect
# ===================================================================

class TestNoOpAction:
    def test_noop_no_remediation_log(self, ready_env: SREEnvironment) -> None:
        obs = ready_env.step(_noop())
        # NoOp should not produce any remediation-related log entries
        remediation_keywords = ["RestartService", "Rollback", "ScaleUp", "ClearCache"]
        for entry in obs.log_buffer:
            for keyword in remediation_keywords:
                assert keyword not in entry, f"NoOp produced remediation log: {entry}"


# ===================================================================
# 13. Correct remediation resolves fault
# ===================================================================

class TestCorrectRemediation:
    def test_rollback_clears_bad_config(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Inject bad_config on db
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")
        assert not env._system._services["db"]["is_healthy"]
        # Apply rollback
        obs = env.step(_rollback("db"))
        # db should recover
        assert obs.health_status["db"] is True

    def test_restart_recovers_degraded_service(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Manually degrade a service
        env._system._services["api"]["is_healthy"] = False
        env._system._services["api"]["cpu"] = 90.0
        obs = env.step(_restart("api"))
        assert obs.health_status["api"] is True
        assert obs.metrics["api"]["cpu"] == pytest.approx(BASELINE_METRICS["cpu"], abs=5.0)


# ===================================================================
# 13b. Rollback clears chaos engine faults and active_incidents
# ===================================================================

class TestRollbackClearsFaults:
    """After a successful rollback that remediates bad_config, the chaos
    engine fault is also cleared and active_incidents becomes empty."""

    def test_rollback_clears_active_incidents(self) -> None:
        """Inject bad_config -> rollback -> active_incidents should be empty."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        # Inject bad_config via chaos engine
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")

        # Verify incident is present
        state = env.state
        assert len(state.active_incidents) > 0
        assert any("bad_config" in inc for inc in state.active_incidents)

        # Apply rollback
        obs = env.step(_rollback("db"))

        # Verify service recovered
        assert obs.health_status["db"] is True

        # Verify active_incidents is empty
        state = env.state
        assert state.active_incidents == [], (
            f"Expected empty active_incidents after rollback, got {state.active_incidents}"
        )

    def test_rollback_clears_chaos_engine_faults(self) -> None:
        """Chaos engine should have no active faults for the service after rollback."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "bad_config", "api")

        # Verify chaos engine tracks the fault
        assert len(env._chaos.get_active_faults()) == 1

        # Apply rollback
        env.step(_rollback("api"))

        # Chaos engine should have no faults
        assert env._chaos.get_active_faults() == []

    def test_rollback_only_clears_target_service_faults(self) -> None:
        """Rollback on one service should not clear faults on other services."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "bad_config", "api")
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")

        # Rollback only api
        env.step(_rollback("api"))

        # api fault should be cleared, db fault should remain
        faults = env._chaos.get_active_faults()
        assert len(faults) == 1
        assert faults[0]["target_service"] == "db"

        # active_incidents should still have the db incident
        state = env.state
        assert len(state.active_incidents) > 0
        assert any("db" in inc for inc in state.active_incidents)

    def test_full_rollback_lifecycle(self) -> None:
        """Full lifecycle: inject bad_config, verify incident appears,
        rollback, verify both chaos engine and system faults cleared."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        # Step 1: Inject bad_config
        env._chaos._inject_specific_fault(env._system, "bad_config", "order")
        assert not env._system._services["order"]["is_healthy"]
        assert env._system._active_faults.get("order") == "bad_config"
        assert len(env._chaos.get_active_faults()) == 1

        # Step 2: Verify incident appears in state
        state = env.state
        assert len(state.active_incidents) > 0

        # Step 3: Rollback
        obs = env.step(_rollback("order"))
        assert obs.health_status["order"] is True

        # Step 4: Verify both system and chaos engine faults cleared
        assert "order" not in env._system._active_faults
        assert env._chaos.get_active_faults() == []
        assert env.state.active_incidents == []


# ===================================================================
# 13c. Rollback only clears bad_config, not other stacked faults
# ===================================================================

class TestRollbackOnlyClearsBadConfig:
    """Rollback should only remove bad_config faults, not memory_leak or
    latent_dependency faults stacked on the same service."""

    def test_rollback_clears_bad_config_keeps_memory_leak(self) -> None:
        """Inject memory_leak + bad_config on same service, rollback ->
        only bad_config cleared, memory_leak persists."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        # Inject both faults on db
        env._chaos._inject_specific_fault(env._system, "memory_leak", "db")
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")

        # Verify both faults are active
        faults = env._chaos.get_active_faults()
        fault_types = {f["fault_type"] for f in faults if f["target_service"] == "db"}
        assert "memory_leak" in fault_types
        assert "bad_config" in fault_types

        # Apply rollback
        obs = env.step(_rollback("db"))

        # bad_config should be cleared (service health restored)
        assert obs.health_status["db"] is True

        # memory_leak should still be active in chaos engine
        remaining_faults = env._chaos.get_active_faults()
        remaining_types = {f["fault_type"] for f in remaining_faults if f["target_service"] == "db"}
        assert "memory_leak" in remaining_types, (
            f"memory_leak should persist after rollback, got: {remaining_faults}"
        )
        assert "bad_config" not in remaining_types, (
            f"bad_config should be cleared by rollback, got: {remaining_faults}"
        )

    def test_rollback_clears_bad_config_keeps_latent_dependency(self) -> None:
        """Inject latent_dependency + bad_config on same service, rollback ->
        only bad_config cleared, latent_dependency persists."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "latent_dependency", "db")
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")

        faults = env._chaos.get_active_faults()
        assert len(faults) == 2

        env.step(_rollback("db"))

        remaining_faults = env._chaos.get_active_faults()
        remaining_types = {f["fault_type"] for f in remaining_faults if f["target_service"] == "db"}
        assert "latent_dependency" in remaining_types, (
            f"latent_dependency should persist after rollback, got: {remaining_faults}"
        )
        assert "bad_config" not in remaining_types

    def test_restart_clears_all_faults_on_service(self) -> None:
        """RestartService should clear ALL faults (memory_leak + bad_config) on the target."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "memory_leak", "db")
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")

        faults = env._chaos.get_active_faults()
        assert len(faults) == 2

        obs = env.step(_restart("db"))

        # All faults should be cleared
        remaining_faults = env._chaos.get_active_faults()
        db_faults = [f for f in remaining_faults if f["target_service"] == "db"]
        assert len(db_faults) == 0, (
            f"RestartService should clear all faults, got: {remaining_faults}"
        )
        assert obs.health_status["db"] is True

    def test_rollback_memory_leak_persists_and_progresses(self) -> None:
        """After rollback clears bad_config, memory_leak continues to progress."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "memory_leak", "db")
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")

        # Rollback clears bad_config only
        env.step(_rollback("db"))

        # Memory should still be progressing from the memory_leak
        mem_after_rollback = env._system._services["db"]["memory"]

        # Take a few more NoOp steps (chaos tick progresses memory_leak)
        for _ in range(3):
            env.step(_noop())

        mem_after_noop_steps = env._system._services["db"]["memory"]
        assert mem_after_noop_steps > mem_after_rollback, (
            f"memory_leak should continue progressing after rollback, "
            f"mem after rollback: {mem_after_rollback}, after steps: {mem_after_noop_steps}"
        )

    def test_rollback_active_incidents_reflects_remaining_fault(self) -> None:
        """After rollback with stacked faults, active_incidents shows remaining fault."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "memory_leak", "db")
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")

        # Before rollback, both incidents should appear
        state = env.state
        assert len(state.active_incidents) >= 2

        # Rollback
        env.step(_rollback("db"))

        # After rollback, memory_leak should still appear in incidents
        state = env.state
        incident_strs = " ".join(state.active_incidents)
        assert "memory_leak" in incident_strs, (
            f"memory_leak incident should persist after rollback, got: {state.active_incidents}"
        )
        assert "bad_config" not in incident_strs, (
            f"bad_config should be cleared from incidents, got: {state.active_incidents}"
        )


# ===================================================================
# 13d. RestartService clears _active_faults and active_incidents
# ===================================================================

class TestRestartClearsActiveFaults:
    """RestartService should clear both chaos engine faults AND system._active_faults
    for the target service, so that active_incidents has no stale entries."""

    def test_restart_clears_bad_config_from_active_faults(self) -> None:
        """Inject bad_config on a service, restart it -> system._active_faults
        should not contain the service anymore."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "bad_config", "db")
        assert env._system._active_faults.get("db") == "bad_config"

        env.step(_restart("db"))

        assert "db" not in env._system._active_faults, (
            f"RestartService should clear db from _active_faults, "
            f"got: {env._system._active_faults}"
        )

    def test_restart_active_incidents_empty_after_bad_config(self) -> None:
        """Inject bad_config on a service, restart it -> active_incidents
        should be empty (no stale entries)."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "bad_config", "order")

        # Verify incident appears before restart
        state = env.state
        assert len(state.active_incidents) > 0
        assert any("bad_config" in inc for inc in state.active_incidents)

        obs = env.step(_restart("order"))

        # After restart, service should be healthy and no stale incidents
        assert obs.health_status["order"] is True
        state = env.state
        assert state.active_incidents == [], (
            f"Expected empty active_incidents after restart, got {state.active_incidents}"
        )

    def test_restart_clears_active_faults_and_chaos_faults(self) -> None:
        """Full lifecycle: inject bad_config via chaos engine, restart,
        verify both system._active_faults and chaos engine faults cleared."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "bad_config", "api")

        # Both tracking systems should have the fault
        assert env._system._active_faults.get("api") == "bad_config"
        assert len(env._chaos.get_active_faults()) == 1

        obs = env.step(_restart("api"))

        # Both should be cleared
        assert "api" not in env._system._active_faults
        assert env._chaos.get_active_faults() == []
        assert obs.health_status["api"] is True
        assert env.state.active_incidents == []

    def test_restart_clears_stacked_faults_from_active_faults(self) -> None:
        """Inject memory_leak + bad_config, restart -> _active_faults cleared,
        chaos engine faults cleared, active_incidents empty."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "memory_leak", "db")
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")

        assert env._system._active_faults.get("db") == "bad_config"
        assert len(env._chaos.get_active_faults()) == 2

        obs = env.step(_restart("db"))

        assert "db" not in env._system._active_faults
        assert env._chaos.get_active_faults() == []
        assert obs.health_status["db"] is True
        assert env.state.active_incidents == []


# ===================================================================
# 14. Mismatched remediation does not resolve fault
# ===================================================================

class TestMismatchedRemediation:
    def test_clearcache_does_not_fix_bad_config(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Inject bad_config on order
        env._chaos._inject_specific_fault(env._system, "bad_config", "order")
        assert not env._system._services["order"]["is_healthy"]
        # Apply ClearCache (wrong action)
        obs = env.step(_clearcache("order"))
        # The service should still be unhealthy (bad_config not cleared by ClearCache)
        assert obs.health_status["order"] is False

    def test_scaleup_does_not_fix_bad_config(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")
        obs = env.step(_scaleup("db"))
        assert obs.health_status["db"] is False


# ===================================================================
# 15. log_buffer capped at 10
# ===================================================================

class TestLogBufferCap:
    def test_log_buffer_never_exceeds_10(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        for i in range(25):
            # Use actions that produce log entries
            obs = env.step(_restart("api"))
            assert len(obs.log_buffer) <= 10

    def test_log_buffer_fifo(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Do many restarts that produce log entries
        for _ in range(15):
            obs = env.step(_restart("api"))
        # Buffer should have 10 entries with latest ones present
        assert len(obs.log_buffer) == 10


# ===================================================================
# 16. Actions scoped to target service only
# ===================================================================

class TestActionScoping:
    def test_restart_scoped_to_target(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        # Degrade db
        env._system._services["db"]["cpu"] = 90.0
        env._system._services["db"]["memory"] = 80.0
        # Save other services' metrics before action
        api_cpu_before = env._system._services["api"]["cpu"]
        order_cpu_before = env._system._services["order"]["cpu"]
        # Restart db
        env.step(_restart("db"))
        # Other services should not have been reset to baseline by restarting db
        # (tick drift is small ±2, so values remain close to their pre-step values)
        api_cpu_after = env._system._services["api"]["cpu"]
        order_cpu_after = env._system._services["order"]["cpu"]
        # They should be within natural drift (~2 per tick) of their prior value
        assert abs(api_cpu_after - api_cpu_before) < 10.0
        assert abs(order_cpu_after - order_cpu_before) < 10.0


# ===================================================================
# 17. SUPPORTS_CONCURRENT_SESSIONS
# ===================================================================

class TestConcurrentSessions:
    def test_supports_concurrent_sessions_true(self) -> None:
        assert SREEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True


# ===================================================================
# 18. render() produces ASCII dashboard
# ===================================================================

class TestRender:
    def test_render_returns_string(self, ready_env: SREEnvironment) -> None:
        output = ready_env.render()
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_contains_service_names(self, ready_env: SREEnvironment) -> None:
        output = ready_env.render()
        for svc in SERVICE_NAMES:
            assert svc.upper() in output.upper() or svc in output

    def test_render_contains_metrics(self, ready_env: SREEnvironment) -> None:
        output = ready_env.render()
        # Should contain some metric labels
        assert "CPU" in output.upper() or "cpu" in output.lower()

    def test_render_fits_terminal(self, ready_env: SREEnvironment) -> None:
        output = ready_env.render()
        for line in output.split("\n"):
            assert len(line) <= 120, f"Line too long ({len(line)} chars): {line[:80]}..."

    def test_render_only_printable_ascii(self, ready_env: SREEnvironment) -> None:
        output = ready_env.render()
        for ch in output:
            assert ch == "\n" or (32 <= ord(ch) <= 126), f"Non-printable char: {ord(ch)}"

    def test_render_shows_reward_info(self) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        env.step(_noop())
        output = env.render()
        # Should contain reward information
        assert "reward" in output.lower() or "Reward" in output


# ===================================================================
# 19. Episode lifecycle integration test
# ===================================================================

class TestEpisodeLifecycle:
    def test_full_episode_to_max_steps(self) -> None:
        max_s = 20
        env = SREEnvironment(max_steps=max_s)
        obs = env.reset(seed=42)
        assert obs.done is False

        for i in range(max_s):
            obs = env.step(_noop())
            assert isinstance(obs, SREObservation)
            if obs.done:
                assert env.state.step_count <= max_s
                break

    def test_multiple_episodes(self) -> None:
        env = SREEnvironment(max_steps=10)
        for episode in range(3):
            obs = env.reset(seed=episode)
            assert obs.done is False
            for _ in range(10):
                obs = env.step(_noop())
                if obs.done:
                    break
            assert obs.done is True

    def test_reset_after_done_starts_fresh(self) -> None:
        env = SREEnvironment(max_steps=5)
        env.reset(seed=42)
        for _ in range(5):
            obs = env.step(_noop())
        assert obs.done is True
        # Reset should work fine
        obs = env.reset(seed=99)
        assert obs.done is False
        assert env.state.step_count == 0


# ===================================================================
# 20. Custom episode_id
# ===================================================================

class TestEpisodeId:
    def test_custom_episode_id(self, env: SREEnvironment) -> None:
        custom_id = "my-episode-123"
        env.reset(seed=42, episode_id=custom_id)
        assert env.state.episode_id == custom_id

    def test_auto_generated_episode_id(self, env: SREEnvironment) -> None:
        env.reset(seed=42)
        assert env.state.episode_id is not None
        assert len(env.state.episode_id) > 0


# ===================================================================
# 21. All action types work through step()
# ===================================================================

class TestAllActionTypes:
    @pytest.mark.parametrize("action_type", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("target", ["api", "order", "db"])
    def test_all_valid_actions(self, action_type: int, target: str) -> None:
        env = SREEnvironment()
        env.reset(seed=42)
        action = SREAction(action_type=action_type, target_service=target)
        obs = env.step(action)
        assert isinstance(obs, SREObservation)


# ===================================================================
# 22. Latent dependency timeout signaling through environment
#     (VAL-CHAOS-002)
# ===================================================================

class TestLatentDependencyTimeoutViaEnv:
    """Integration: latent_dependency fault -> threshold exceeded -> timeout alerts."""

    def test_timeout_alerts_appear_in_observation(self) -> None:
        """After enough steps with latent_dependency, observation has timeout alerts."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        # Inject latent_dependency on db
        env._chaos._inject_specific_fault(env._system, "latent_dependency", "db")

        # Step until threshold is exceeded (baseline=50, +20/tick => ~23 ticks)
        obs = None
        for _ in range(30):
            obs = env.step(_noop())

        assert obs is not None
        timeout_alerts = [a for a in obs.active_alerts if "Timeout:" in a]
        assert len(timeout_alerts) >= 1, (
            f"Expected timeout alert in observation, got: {obs.active_alerts}"
        )

    def test_upstream_timeout_alerts_in_observation(self) -> None:
        """After many steps, upstream services also show timeout alerts."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)

        env._chaos._inject_specific_fault(env._system, "latent_dependency", "db")

        obs = None
        for _ in range(60):
            obs = env.step(_noop())

        assert obs is not None
        # Look for order or api timeout alerts
        upstream_alerts = [
            a for a in obs.active_alerts
            if "Timeout:" in a and ("order" in a or "api" in a)
        ]
        assert len(upstream_alerts) >= 1, (
            f"Expected upstream timeout alerts, got: {obs.active_alerts}"
        )


# ===================================================================
# 23. Latent dependency recovery through environment
#     (VAL-CHAOS-007)
# ===================================================================

class TestLatentDependencyRecoveryViaEnv:
    """Integration: latent_dependency on db -> step -> restart db ->
    upstream latencies return to near-baseline."""

    def test_restart_db_normalises_upstream_latencies(self) -> None:
        """Restarting the root cause of latent_dependency normalises upstream."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)
        baseline = BASELINE_METRICS["latency"]

        env._chaos._inject_specific_fault(env._system, "latent_dependency", "db")

        # Step to build up cascaded latency
        for _ in range(10):
            env.step(_noop())

        # Verify elevated upstream latency
        assert env._system._services["order"]["latency"] > baseline + 10.0
        assert env._system._services["api"]["latency"] > baseline + 5.0

        # Restart db to remediate
        obs = env.step(_restart("db"))

        # After restart, upstream latencies should be near baseline
        # (Note: natural tick drift ±5 applied after restart)
        order_lat = obs.metrics["order"]["latency"]
        api_lat = obs.metrics["api"]["latency"]
        assert order_lat < baseline + 20.0, (
            f"order latency should be near baseline after recovery, got {order_lat}"
        )
        assert api_lat < baseline + 20.0, (
            f"api latency should be near baseline after recovery, got {api_lat}"
        )

    def test_rollback_does_not_clear_latent_dependency(self) -> None:
        """Rollback does NOT clear latent_dependency faults — only bad_config.
        Latent dependency requires RestartService to remediate."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)
        baseline = BASELINE_METRICS["latency"]

        env._chaos._inject_specific_fault(env._system, "latent_dependency", "db")

        for _ in range(10):
            env.step(_noop())

        assert env._system._services["order"]["latency"] > baseline + 10.0

        obs = env.step(_rollback("db"))

        # Rollback should NOT clear latent_dependency, so upstream latencies
        # remain elevated (latent_dependency continues progressing).
        order_lat = obs.metrics["order"]["latency"]
        assert order_lat > baseline + 10.0, (
            f"order latency should stay elevated after rollback (latent_dependency not cleared), "
            f"got {order_lat}"
        )

        # latent_dependency should still be active in chaos engine
        remaining = env._chaos.get_active_faults()
        assert any(
            f["fault_type"] == "latent_dependency" and f["target_service"] == "db"
            for f in remaining
        ), f"latent_dependency should persist after rollback, got: {remaining}"

    def test_full_lifecycle_fault_cascade_recovery(self) -> None:
        """Full lifecycle: inject latent_dependency on db, observe cascade,
        restart db, verify all tiers return to baseline."""
        env = SREEnvironment(fault_probability=0.0)
        env.reset(seed=42)
        baseline = BASELINE_METRICS["latency"]

        # Phase 1: Inject and cascade
        env._chaos._inject_specific_fault(env._system, "latent_dependency", "db")

        latencies_over_time = {"db": [], "order": [], "api": []}
        for _ in range(15):
            obs = env.step(_noop())
            for svc in SERVICE_NAMES:
                latencies_over_time[svc].append(obs.metrics[svc]["latency"])

        # Verify all tiers degraded
        for svc in SERVICE_NAMES:
            assert latencies_over_time[svc][-1] > baseline, (
                f"{svc} should be degraded, last latency={latencies_over_time[svc][-1]}"
            )

        # Phase 2: Remediate db
        obs = env.step(_restart("db"))

        # Phase 3: Verify recovery
        # db restarted to baseline (then drift applied)
        db_lat = obs.metrics["db"]["latency"]
        order_lat = obs.metrics["order"]["latency"]
        api_lat = obs.metrics["api"]["latency"]

        assert db_lat < baseline + 15.0, (
            f"db should be near baseline after restart, got {db_lat}"
        )
        assert order_lat < baseline + 20.0, (
            f"order should normalise after db restart, got {order_lat}"
        )
        assert api_lat < baseline + 20.0, (
            f"api should normalise after db restart, got {api_lat}"
        )
