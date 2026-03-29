"""Tests for RewardCalculator — TDD-first covering all 5 reward components.

Covers:
  - Each component independently
  - Boundary conditions (latency exactly 100, all healthy, all down)
  - Stacking scenario (multiple penalties in one step)
  - Ideal conditions (all healthy + NoOp = +1.0)
  - Validation contract assertions: VAL-REW-001 through VAL-REW-006
"""

from __future__ import annotations

import pytest

from sre_agent_sandbox.models import SREAction
from sre_agent_sandbox.reward import RewardCalculator
from sre_agent_sandbox.simulated_system import SimulatedSystem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_system(
    *,
    healthy: dict[str, bool] | None = None,
    down: dict[str, bool] | None = None,
    latencies: dict[str, float] | None = None,
) -> SimulatedSystem:
    """Create a SimulatedSystem and optionally override health/latency."""
    system = SimulatedSystem()
    system.reset(seed=42)

    if healthy is not None:
        for svc, is_healthy in healthy.items():
            system._services[svc]["is_healthy"] = is_healthy

    if down is not None:
        for svc, is_down in down.items():
            system._services[svc]["is_down"] = is_down
            if is_down:
                system._services[svc]["is_healthy"] = False

    if latencies is not None:
        for svc, lat in latencies.items():
            system._services[svc]["latency"] = lat

    return system


def _noop_action() -> SREAction:
    return SREAction(action_type=0, target_service="api")


def _restart_action(target: str = "api") -> SREAction:
    return SREAction(action_type=1, target_service=target)


def _rollback_action(target: str = "api") -> SREAction:
    return SREAction(action_type=2, target_service=target)


def _scaleup_action(target: str = "api") -> SREAction:
    return SREAction(action_type=3, target_service=target)


def _clearcache_action(target: str = "api") -> SREAction:
    return SREAction(action_type=4, target_service=target)


# ===================================================================
# Test: RewardCalculator construction
# ===================================================================


class TestRewardCalculatorConstruction:
    """Basic construction and interface tests."""

    def test_calculator_creates_successfully(self) -> None:
        calc = RewardCalculator()
        assert calc is not None

    def test_calculate_returns_tuple(self) -> None:
        calc = RewardCalculator()
        system = _make_system()
        action = _noop_action()
        result = calc.calculate(system, action)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_calculate_returns_float_and_dict(self) -> None:
        calc = RewardCalculator()
        system = _make_system()
        action = _noop_action()
        total, breakdown = calc.calculate(system, action)
        assert isinstance(total, float)
        assert isinstance(breakdown, dict)

    def test_breakdown_has_all_five_components(self) -> None:
        calc = RewardCalculator()
        system = _make_system()
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        expected_keys = {"availability", "latency", "downtime", "efficiency", "safety"}
        assert set(breakdown.keys()) == expected_keys


# ===================================================================
# Test: Availability component (VAL-REW-001)
# ===================================================================


class TestAvailabilityComponent:
    """Availability: +1.0 when ALL services healthy, 0.0 otherwise."""

    def test_all_healthy_gives_plus_one(self) -> None:
        """VAL-REW-001: All healthy -> availability = +1.0."""
        calc = RewardCalculator()
        system = _make_system()
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["availability"] == 1.0

    def test_one_unhealthy_gives_zero(self) -> None:
        """VAL-REW-001: One unhealthy -> availability = 0.0."""
        calc = RewardCalculator()
        system = _make_system(healthy={"api": False})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["availability"] == 0.0

    def test_two_unhealthy_gives_zero(self) -> None:
        calc = RewardCalculator()
        system = _make_system(healthy={"api": False, "order": False})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["availability"] == 0.0

    def test_all_unhealthy_gives_zero(self) -> None:
        calc = RewardCalculator()
        system = _make_system(healthy={"api": False, "order": False, "db": False})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["availability"] == 0.0

    def test_down_service_counts_as_unhealthy(self) -> None:
        """A service that is_down is also not healthy -> availability 0.0."""
        calc = RewardCalculator()
        system = _make_system(down={"db": True})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["availability"] == 0.0


# ===================================================================
# Test: Latency penalty component (VAL-REW-002)
# ===================================================================


class TestLatencyPenaltyComponent:
    """Latency: -0.01 * (avg_latency - 100) when > 100ms, else 0.0."""

    def test_below_100ms_no_penalty(self) -> None:
        """avg_latency < 100 -> 0.0."""
        calc = RewardCalculator()
        # Baseline latency is 50ms for all services -> avg = 50
        system = _make_system()
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["latency"] == 0.0

    def test_exactly_100ms_no_penalty(self) -> None:
        """VAL-REW-002 boundary: exactly 100ms -> penalty = 0.0."""
        calc = RewardCalculator()
        system = _make_system(latencies={"api": 100.0, "order": 100.0, "db": 100.0})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["latency"] == 0.0

    def test_above_100ms_penalty(self) -> None:
        """avg_latency > 100 -> -0.01 * (avg - 100)."""
        calc = RewardCalculator()
        # avg = (200 + 200 + 200) / 3 = 200
        system = _make_system(latencies={"api": 200.0, "order": 200.0, "db": 200.0})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        expected = -0.01 * (200.0 - 100.0)  # = -1.0
        assert breakdown["latency"] == pytest.approx(expected)

    def test_mixed_latency_avg_above_100(self) -> None:
        """Mixed latencies: avg = (50 + 150 + 200) / 3 = 133.33."""
        calc = RewardCalculator()
        system = _make_system(latencies={"api": 50.0, "order": 150.0, "db": 200.0})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        avg = (50.0 + 150.0 + 200.0) / 3.0
        expected = -0.01 * (avg - 100.0)
        assert breakdown["latency"] == pytest.approx(expected)

    def test_mixed_latency_avg_below_100(self) -> None:
        """Mixed latencies where avg < 100."""
        calc = RewardCalculator()
        system = _make_system(latencies={"api": 50.0, "order": 80.0, "db": 90.0})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        # avg = (50 + 80 + 90) / 3 = 73.33 < 100
        assert breakdown["latency"] == 0.0

    def test_one_very_high_latency(self) -> None:
        """One service has very high latency."""
        calc = RewardCalculator()
        system = _make_system(latencies={"api": 50.0, "order": 50.0, "db": 500.0})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        avg = (50.0 + 50.0 + 500.0) / 3.0  # = 200
        expected = -0.01 * (avg - 100.0)
        assert breakdown["latency"] == pytest.approx(expected)

    def test_latency_just_above_100(self) -> None:
        """Avg latency 100.01 -> small penalty."""
        calc = RewardCalculator()
        # Each service at 100.01 -> avg = 100.01
        system = _make_system(latencies={"api": 100.01, "order": 100.01, "db": 100.01})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        expected = -0.01 * (100.01 - 100.0)  # = -0.0001
        assert breakdown["latency"] == pytest.approx(expected)


# ===================================================================
# Test: Downtime penalty component (VAL-REW-003)
# ===================================================================


class TestDowntimePenaltyComponent:
    """Downtime: -5.0 if ANY service is_down, once per step."""

    def test_no_services_down_no_penalty(self) -> None:
        calc = RewardCalculator()
        system = _make_system()
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["downtime"] == 0.0

    def test_one_service_down_penalty(self) -> None:
        """VAL-REW-003: One down -> -5.0."""
        calc = RewardCalculator()
        system = _make_system(down={"api": True})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["downtime"] == -5.0

    def test_two_services_down_still_minus_five(self) -> None:
        """VAL-REW-003: Two down -> still -5.0 (once per step)."""
        calc = RewardCalculator()
        system = _make_system(down={"api": True, "order": True})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["downtime"] == -5.0

    def test_all_services_down_still_minus_five(self) -> None:
        """VAL-REW-003: All three down -> still -5.0 (once per step)."""
        calc = RewardCalculator()
        system = _make_system(down={"api": True, "order": True, "db": True})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["downtime"] == -5.0


# ===================================================================
# Test: Efficiency penalty component (VAL-REW-004)
# ===================================================================


class TestEfficiencyPenaltyComponent:
    """Efficiency: -0.1 for non-NoOp, 0.0 for NoOp."""

    def test_noop_no_penalty(self) -> None:
        """VAL-REW-004: NoOp -> 0.0 efficiency."""
        calc = RewardCalculator()
        system = _make_system()
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["efficiency"] == 0.0

    def test_restart_penalty(self) -> None:
        """VAL-REW-004: RestartService -> -0.1."""
        calc = RewardCalculator()
        # Make api unhealthy so we don't also trigger safety penalty
        system = _make_system(healthy={"api": False})
        action = _restart_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["efficiency"] == -0.1

    def test_rollback_penalty(self) -> None:
        """VAL-REW-004: Rollback -> -0.1."""
        calc = RewardCalculator()
        system = _make_system()
        action = _rollback_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["efficiency"] == -0.1

    def test_scaleup_penalty(self) -> None:
        """VAL-REW-004: ScaleUp -> -0.1."""
        calc = RewardCalculator()
        system = _make_system()
        action = _scaleup_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["efficiency"] == -0.1

    def test_clearcache_penalty(self) -> None:
        """VAL-REW-004: ClearCache -> -0.1."""
        calc = RewardCalculator()
        system = _make_system()
        action = _clearcache_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["efficiency"] == -0.1


# ===================================================================
# Test: Safety penalty component (VAL-REW-005)
# ===================================================================


class TestSafetyPenaltyComponent:
    """Safety: -10.0 if RestartService AND target is_healthy."""

    def test_restart_healthy_service_penalty(self) -> None:
        """VAL-REW-005: Restart healthy -> -10.0."""
        calc = RewardCalculator()
        system = _make_system()  # all healthy
        action = _restart_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["safety"] == -10.0

    def test_restart_unhealthy_service_no_penalty(self) -> None:
        """VAL-REW-005: Restart unhealthy -> 0.0 safety."""
        calc = RewardCalculator()
        system = _make_system(healthy={"api": False})
        action = _restart_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["safety"] == 0.0

    def test_restart_down_service_no_penalty(self) -> None:
        """Restart a down service -> 0.0 safety (not healthy)."""
        calc = RewardCalculator()
        system = _make_system(down={"api": True})
        action = _restart_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["safety"] == 0.0

    def test_noop_on_healthy_no_safety_penalty(self) -> None:
        """NoOp never triggers safety, even with healthy services."""
        calc = RewardCalculator()
        system = _make_system()
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["safety"] == 0.0

    def test_rollback_healthy_no_safety_penalty(self) -> None:
        """Rollback on healthy service -> 0.0 safety (only restart triggers it)."""
        calc = RewardCalculator()
        system = _make_system()
        action = _rollback_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["safety"] == 0.0

    def test_scaleup_healthy_no_safety_penalty(self) -> None:
        """ScaleUp on healthy service -> 0.0 safety."""
        calc = RewardCalculator()
        system = _make_system()
        action = _scaleup_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["safety"] == 0.0

    def test_clearcache_healthy_no_safety_penalty(self) -> None:
        """ClearCache on healthy service -> 0.0 safety."""
        calc = RewardCalculator()
        system = _make_system()
        action = _clearcache_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["safety"] == 0.0

    def test_restart_each_healthy_service(self) -> None:
        """Safety penalty applies regardless of which healthy service is restarted."""
        calc = RewardCalculator()
        for svc in ["api", "order", "db"]:
            system = _make_system()  # all healthy
            action = _restart_action(svc)
            _, breakdown = calc.calculate(system, action)
            assert breakdown["safety"] == -10.0, f"Expected -10.0 for restarting healthy {svc}"


# ===================================================================
# Test: Total reward sum (VAL-REW-006)
# ===================================================================


class TestTotalRewardSum:
    """Total reward = sum of all 5 components."""

    def test_total_equals_sum_of_components(self) -> None:
        """VAL-REW-006: Total = sum of breakdown values."""
        calc = RewardCalculator()
        system = _make_system()
        action = _noop_action()
        total, breakdown = calc.calculate(system, action)
        expected_total = sum(breakdown.values())
        assert total == pytest.approx(expected_total)

    def test_total_equals_sum_with_penalties(self) -> None:
        """VAL-REW-006: Stacking multiple penalties."""
        calc = RewardCalculator()
        # All services down + high latency + restart action on healthy "order"
        # But wait, if all services are down, none are healthy.
        # Let's use: api down, high latency, restart on healthy db
        system = _make_system(
            down={"api": True},
            latencies={"api": 200.0, "order": 200.0, "db": 200.0},
        )
        action = _restart_action("db")  # db is still healthy
        total, breakdown = calc.calculate(system, action)
        expected_total = sum(breakdown.values())
        assert total == pytest.approx(expected_total)


# ===================================================================
# Test: Ideal conditions (VAL-REW-001 + all healthy + NoOp)
# ===================================================================


class TestIdealConditions:
    """Ideal: all healthy, NoOp, low latency -> total = +1.0."""

    def test_ideal_conditions_total_reward(self) -> None:
        """All healthy + NoOp + latency < 100 -> exactly +1.0."""
        calc = RewardCalculator()
        system = _make_system()  # all healthy, baseline latency=50
        action = _noop_action()
        total, breakdown = calc.calculate(system, action)

        assert breakdown["availability"] == 1.0
        assert breakdown["latency"] == 0.0
        assert breakdown["downtime"] == 0.0
        assert breakdown["efficiency"] == 0.0
        assert breakdown["safety"] == 0.0
        assert total == pytest.approx(1.0)

    def test_ideal_conditions_each_service_target(self) -> None:
        """NoOp with any target service should give +1.0 under ideal conditions."""
        calc = RewardCalculator()
        for svc in ["api", "order", "db"]:
            system = _make_system()
            action = SREAction(action_type=0, target_service=svc)
            total, _ = calc.calculate(system, action)
            assert total == pytest.approx(1.0), f"Expected +1.0 for NoOp on {svc}"


# ===================================================================
# Test: Stacking scenario (multiple penalties in one step)
# ===================================================================


class TestStackingScenario:
    """Multiple penalties stacking in a single step."""

    def test_availability_zero_plus_downtime_plus_latency(self) -> None:
        """One service down, high latency, NoOp."""
        calc = RewardCalculator()
        system = _make_system(
            down={"api": True},
            latencies={"api": 300.0, "order": 300.0, "db": 300.0},
        )
        action = _noop_action()
        total, breakdown = calc.calculate(system, action)

        # availability: 0.0 (api unhealthy)
        assert breakdown["availability"] == 0.0
        # latency: -0.01 * (300 - 100) = -2.0
        assert breakdown["latency"] == pytest.approx(-2.0)
        # downtime: -5.0 (api down)
        assert breakdown["downtime"] == -5.0
        # efficiency: 0.0 (NoOp)
        assert breakdown["efficiency"] == 0.0
        # safety: 0.0 (NoOp)
        assert breakdown["safety"] == 0.0
        # total: 0.0 - 2.0 - 5.0 = -7.0
        assert total == pytest.approx(-7.0)

    def test_all_penalties_stacked(self) -> None:
        """All 5 components contribute non-trivially in one step."""
        calc = RewardCalculator()
        # api is down, order is healthy but has high latency, db is healthy
        # Use restart on db (healthy) -> safety penalty + efficiency penalty
        system = _make_system(
            down={"api": True},
            latencies={"api": 200.0, "order": 200.0, "db": 200.0},
        )
        action = _restart_action("db")  # db is healthy -> safety

        total, breakdown = calc.calculate(system, action)

        # availability: 0.0 (api is unhealthy/down)
        assert breakdown["availability"] == 0.0
        # latency: -0.01 * (200 - 100) = -1.0
        assert breakdown["latency"] == pytest.approx(-1.0)
        # downtime: -5.0 (api is down)
        assert breakdown["downtime"] == -5.0
        # efficiency: -0.1 (RestartService is non-NoOp)
        assert breakdown["efficiency"] == -0.1
        # safety: -10.0 (restarting healthy db)
        assert breakdown["safety"] == -10.0
        # total: 0.0 - 1.0 - 5.0 - 0.1 - 10.0 = -16.1
        assert total == pytest.approx(-16.1)

    def test_restart_unhealthy_with_downtime_and_high_latency(self) -> None:
        """Restart an unhealthy service: efficiency penalty but no safety."""
        calc = RewardCalculator()
        system = _make_system(
            down={"order": True},
            latencies={"api": 150.0, "order": 150.0, "db": 150.0},
        )
        action = _restart_action("order")  # order is down (unhealthy)

        total, breakdown = calc.calculate(system, action)

        # availability: 0.0
        assert breakdown["availability"] == 0.0
        # latency: -0.01 * (150 - 100) = -0.5
        assert breakdown["latency"] == pytest.approx(-0.5)
        # downtime: -5.0
        assert breakdown["downtime"] == -5.0
        # efficiency: -0.1
        assert breakdown["efficiency"] == -0.1
        # safety: 0.0 (order is not healthy)
        assert breakdown["safety"] == 0.0
        # total: 0.0 - 0.5 - 5.0 - 0.1 + 0.0 = -5.6
        assert total == pytest.approx(-5.6)


# ===================================================================
# Test: Boundary conditions
# ===================================================================


class TestBoundaryConditions:
    """Boundary and edge-case tests."""

    def test_all_services_down_max_downtime(self) -> None:
        """All three down -> downtime is still only -5.0."""
        calc = RewardCalculator()
        system = _make_system(down={"api": True, "order": True, "db": True})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["downtime"] == -5.0

    def test_latency_exactly_100_boundary(self) -> None:
        """Exactly 100ms avg latency -> 0.0 penalty."""
        calc = RewardCalculator()
        system = _make_system(latencies={"api": 100.0, "order": 100.0, "db": 100.0})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["latency"] == 0.0

    def test_zero_latency(self) -> None:
        """All latencies = 0 -> 0.0 penalty."""
        calc = RewardCalculator()
        system = _make_system(latencies={"api": 0.0, "order": 0.0, "db": 0.0})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        assert breakdown["latency"] == 0.0

    def test_very_high_latency(self) -> None:
        """Very high latency produces large penalty."""
        calc = RewardCalculator()
        system = _make_system(latencies={"api": 10000.0, "order": 10000.0, "db": 10000.0})
        action = _noop_action()
        _, breakdown = calc.calculate(system, action)
        expected = -0.01 * (10000.0 - 100.0)  # = -99.0
        assert breakdown["latency"] == pytest.approx(expected)

    def test_restart_service_that_is_not_target_healthy(self) -> None:
        """Restart targets a specific service; safety checks that specific target."""
        calc = RewardCalculator()
        # api unhealthy, but we restart healthy "db"
        system = _make_system(healthy={"api": False})
        action = _restart_action("db")
        _, breakdown = calc.calculate(system, action)
        # db is healthy -> safety penalty
        assert breakdown["safety"] == -10.0

    def test_restart_unhealthy_target_while_others_healthy(self) -> None:
        """Restart an unhealthy target while others are healthy -> no safety penalty."""
        calc = RewardCalculator()
        system = _make_system(healthy={"api": False})
        action = _restart_action("api")
        _, breakdown = calc.calculate(system, action)
        assert breakdown["safety"] == 0.0


# ===================================================================
# Test: Cross-area validation (VAL-CROSS-003)
# ===================================================================


class TestCrossArea:
    """Cross-area: incorrect action on healthy service triggers safety penalty."""

    def test_restart_healthy_worse_than_noop(self) -> None:
        """VAL-CROSS-003: Restart healthy -> lower total than NoOp."""
        calc = RewardCalculator()
        system = _make_system()  # all healthy

        noop_total, _ = calc.calculate(system, _noop_action())
        restart_total, _ = calc.calculate(system, _restart_action("api"))

        assert restart_total < noop_total
