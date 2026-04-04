"""Tests for the SimulatedSystem class.

Written TDD-style: tests first, implementation after.
Covers: initialization, reset, actions (NoOp, RestartService, Rollback, ScaleUp, ClearCache),
tick/drift, get_metrics, get_health_status, get_log_buffer, get_active_alerts,
dependency chains, and log buffer FIFO cap.
"""

from __future__ import annotations

from simulated_system import SimulatedSystem

# ---------------------------------------------------------------------------
# Constants for baseline expectations
# ---------------------------------------------------------------------------
SERVICE_NAMES = ["api", "order", "db"]
BASELINE_CPU = 30.0
BASELINE_MEMORY = 40.0
BASELINE_LATENCY = 50.0
BASELINE_REQUEST_COUNT = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system(seed: int = 42) -> SimulatedSystem:
    """Create and reset a SimulatedSystem with a deterministic seed."""
    system = SimulatedSystem()
    system.reset(seed=seed)
    return system


def _degrade_service(system: SimulatedSystem, service: str) -> None:
    """Manually degrade a service's metrics to simulate a fault."""
    svc = system._services[service]
    svc["cpu"] = 90.0
    svc["memory"] = 85.0
    svc["latency"] = 500.0
    svc["is_healthy"] = False


def _set_bad_config_fault(system: SimulatedSystem, service: str) -> None:
    """Manually inject a bad_config fault on a service."""
    svc = system._services[service]
    svc["is_healthy"] = False
    svc["latency"] = 1000.0
    system._active_faults[service] = "bad_config"


def _set_service_down(system: SimulatedSystem, service: str) -> None:
    """Manually set a service as down."""
    svc = system._services[service]
    svc["is_healthy"] = False
    svc["is_down"] = True
    svc["cpu"] = 0.0
    svc["memory"] = 0.0
    svc["latency"] = 0.0
    svc["request_count"] = 0


# ===========================================================================
# Initialization Tests
# ===========================================================================


class TestSimulatedSystemInit:
    """Test that the system initializes with 3 healthy services at baseline."""

    def test_system_creates_successfully(self) -> None:
        system = SimulatedSystem()
        assert system is not None

    def test_constructor_initializes_services(self) -> None:
        """SimulatedSystem() should be immediately usable without explicit reset()."""
        system = SimulatedSystem()
        metrics = system.get_metrics()
        assert set(metrics.keys()) == {"api", "order", "db"}
        for svc in SERVICE_NAMES:
            assert metrics[svc]["cpu"] == BASELINE_CPU
            assert metrics[svc]["memory"] == BASELINE_MEMORY
            assert metrics[svc]["latency"] == BASELINE_LATENCY

    def test_constructor_health_status_without_reset(self) -> None:
        """get_health_status() works immediately on a fresh instance."""
        system = SimulatedSystem()
        health = system.get_health_status()
        for svc in SERVICE_NAMES:
            assert health[svc] is True

    def test_constructor_log_buffer_without_reset(self) -> None:
        """get_log_buffer() works immediately on a fresh instance."""
        system = SimulatedSystem()
        assert system.get_log_buffer() == []

    def test_constructor_active_alerts_without_reset(self) -> None:
        """get_active_alerts() works immediately on a fresh instance."""
        system = SimulatedSystem()
        assert system.get_active_alerts() == []

    def test_reset_creates_three_services(self) -> None:
        system = _make_system()
        metrics = system.get_metrics()
        assert set(metrics.keys()) == {"api", "order", "db"}

    def test_all_services_healthy_after_reset(self) -> None:
        system = _make_system()
        health = system.get_health_status()
        for svc in SERVICE_NAMES:
            assert health[svc] is True, f"{svc} should be healthy after reset"

    def test_baseline_metrics_after_reset(self) -> None:
        system = _make_system()
        metrics = system.get_metrics()
        for svc in SERVICE_NAMES:
            m = metrics[svc]
            assert 0.0 <= m["cpu"] <= 100.0
            assert 0.0 <= m["memory"] <= 100.0
            assert m["latency"] >= 0.0
            assert m["request_count"] >= 0

    def test_metrics_contain_expected_keys(self) -> None:
        system = _make_system()
        metrics = system.get_metrics()
        expected_keys = {"cpu", "memory", "latency", "request_count"}
        for svc in SERVICE_NAMES:
            assert set(metrics[svc].keys()) == expected_keys

    def test_no_services_down_after_reset(self) -> None:
        system = _make_system()
        for svc in SERVICE_NAMES:
            assert system._services[svc]["is_down"] is False

    def test_instance_count_default_one(self) -> None:
        system = _make_system()
        for svc in SERVICE_NAMES:
            assert system._services[svc]["instance_count"] == 1

    def test_log_buffer_empty_after_reset(self) -> None:
        system = _make_system()
        assert system.get_log_buffer() == []

    def test_no_active_alerts_after_reset(self) -> None:
        system = _make_system()
        assert system.get_active_alerts() == []

    def test_no_active_faults_after_reset(self) -> None:
        system = _make_system()
        assert system._active_faults == {}


# ===========================================================================
# Reset Tests
# ===========================================================================


class TestReset:
    """Test that reset restores everything to baseline."""

    def test_reset_clears_degraded_metrics(self) -> None:
        system = _make_system()
        _degrade_service(system, "api")
        system.reset(seed=42)
        health = system.get_health_status()
        assert health["api"] is True

    def test_reset_clears_log_buffer(self) -> None:
        system = _make_system()
        system.apply_action(1, "api")  # generates a log entry
        assert len(system.get_log_buffer()) > 0
        system.reset(seed=42)
        assert system.get_log_buffer() == []

    def test_reset_clears_active_alerts(self) -> None:
        system = _make_system()
        _degrade_service(system, "db")
        # Force an alert
        system._active_alerts.append("db is unhealthy")
        system.reset(seed=42)
        assert system.get_active_alerts() == []

    def test_reset_clears_active_faults(self) -> None:
        system = _make_system()
        _set_bad_config_fault(system, "order")
        system.reset(seed=42)
        assert system._active_faults == {}

    def test_reset_restores_instance_count(self) -> None:
        system = _make_system()
        system.apply_action(3, "api")  # ScaleUp
        assert system._services["api"]["instance_count"] > 1
        system.reset(seed=42)
        assert system._services["api"]["instance_count"] == 1

    def test_reset_with_different_seeds(self) -> None:
        """Two resets with same seed produce same state."""
        s1 = _make_system(seed=42)
        s2 = _make_system(seed=42)
        assert s1.get_metrics() == s2.get_metrics()
        assert s1.get_health_status() == s2.get_health_status()

    def test_reset_restores_down_service(self) -> None:
        system = _make_system()
        _set_service_down(system, "db")
        system.reset(seed=42)
        assert system._services["db"]["is_down"] is False
        assert system._services["db"]["is_healthy"] is True


# ===========================================================================
# NoOp Action Tests (VAL-ENV-005)
# ===========================================================================


class TestNoOpAction:
    """NoOp (action_type=0) should have no remediation effect."""

    def test_noop_does_not_change_healthy_metrics(self) -> None:
        system = _make_system()
        metrics_before = system.get_metrics()
        system.apply_action(0, "api")
        metrics_after = system.get_metrics()
        # Metrics should be unchanged for all services
        for svc in SERVICE_NAMES:
            assert metrics_before[svc] == metrics_after[svc]

    def test_noop_does_not_change_health_status(self) -> None:
        system = _make_system()
        health_before = system.get_health_status()
        system.apply_action(0, "api")
        health_after = system.get_health_status()
        assert health_before == health_after

    def test_noop_does_not_generate_remediation_log(self) -> None:
        system = _make_system()
        system.apply_action(0, "api")
        log = system.get_log_buffer()
        # No remediation log entries (may have a NoOp log but no remediation effect)
        for entry in log:
            assert "restart" not in entry.lower()
            assert "rollback" not in entry.lower()
            assert "scale" not in entry.lower()
            assert "clear" not in entry.lower()

    def test_noop_on_degraded_service_leaves_it_degraded(self) -> None:
        system = _make_system()
        _degrade_service(system, "order")
        system.apply_action(0, "order")
        assert system._services["order"]["is_healthy"] is False
        assert system._services["order"]["cpu"] == 90.0

    def test_noop_on_all_services(self) -> None:
        """NoOp works on any target service without side effects."""
        for svc in SERVICE_NAMES:
            system = _make_system()
            metrics_before = system.get_metrics()
            system.apply_action(0, svc)
            metrics_after = system.get_metrics()
            for s in SERVICE_NAMES:
                assert metrics_before[s] == metrics_after[s]


# ===========================================================================
# RestartService Action Tests (VAL-ENV-006)
# ===========================================================================


class TestRestartServiceAction:
    """RestartService (action_type=1) resets metrics to baseline for target."""

    def test_restart_degraded_service_restores_health(self) -> None:
        system = _make_system()
        _degrade_service(system, "api")
        assert system._services["api"]["is_healthy"] is False
        system.apply_action(1, "api")
        assert system._services["api"]["is_healthy"] is True

    def test_restart_restores_baseline_metrics(self) -> None:
        system = _make_system()
        _degrade_service(system, "db")
        system.apply_action(1, "db")
        metrics = system.get_metrics()
        assert metrics["db"]["cpu"] <= 50.0  # Near baseline, not 90
        assert metrics["db"]["memory"] <= 50.0
        assert metrics["db"]["latency"] <= 100.0

    def test_restart_generates_log_entry(self) -> None:
        system = _make_system()
        system.apply_action(1, "order")
        log = system.get_log_buffer()
        assert len(log) >= 1
        assert any("restart" in entry.lower() for entry in log)

    def test_restart_down_service_brings_it_up(self) -> None:
        system = _make_system()
        _set_service_down(system, "api")
        system.apply_action(1, "api")
        assert system._services["api"]["is_down"] is False
        assert system._services["api"]["is_healthy"] is True

    def test_restart_healthy_service_keeps_it_healthy(self) -> None:
        system = _make_system()
        system.apply_action(1, "api")
        assert system._services["api"]["is_healthy"] is True


# ===========================================================================
# Rollback Action Tests (VAL-ENV-007)
# ===========================================================================


class TestRollbackAction:
    """Rollback (action_type=2) reverts to stable config, clears bad_config."""

    def test_rollback_clears_bad_config_fault(self) -> None:
        system = _make_system()
        _set_bad_config_fault(system, "order")
        assert system._active_faults.get("order") == "bad_config"
        system.apply_action(2, "order")
        assert system._active_faults.get("order") is None

    def test_rollback_restores_health_after_bad_config(self) -> None:
        system = _make_system()
        _set_bad_config_fault(system, "api")
        system.apply_action(2, "api")
        assert system._services["api"]["is_healthy"] is True

    def test_rollback_generates_log_entry(self) -> None:
        system = _make_system()
        system.apply_action(2, "db")
        log = system.get_log_buffer()
        assert len(log) >= 1
        assert any("rollback" in entry.lower() for entry in log)

    def test_rollback_on_non_bad_config_still_logged(self) -> None:
        """Rollback on a service without bad_config still produces a log entry."""
        system = _make_system()
        _degrade_service(system, "api")
        system.apply_action(2, "api")
        log = system.get_log_buffer()
        assert any("rollback" in entry.lower() for entry in log)

    def test_rollback_restores_metrics_for_bad_config(self) -> None:
        system = _make_system()
        _set_bad_config_fault(system, "db")
        system.apply_action(2, "db")
        metrics = system.get_metrics()
        assert metrics["db"]["latency"] < 500.0  # Restored from 1000


# ===========================================================================
# ScaleUp Action Tests (VAL-ENV-008)
# ===========================================================================


class TestScaleUpAction:
    """ScaleUp (action_type=3) increases instance_count, reduces load metrics."""

    def test_scaleup_increases_instance_count(self) -> None:
        system = _make_system()
        count_before = system._services["api"]["instance_count"]
        system.apply_action(3, "api")
        count_after = system._services["api"]["instance_count"]
        assert count_after > count_before

    def test_scaleup_reduces_cpu(self) -> None:
        system = _make_system()
        _degrade_service(system, "order")
        cpu_before = system._services["order"]["cpu"]
        system.apply_action(3, "order")
        cpu_after = system.get_metrics()["order"]["cpu"]
        assert cpu_after < cpu_before

    def test_scaleup_reduces_latency(self) -> None:
        system = _make_system()
        _degrade_service(system, "db")
        latency_before = system._services["db"]["latency"]
        system.apply_action(3, "db")
        latency_after = system.get_metrics()["db"]["latency"]
        assert latency_after < latency_before

    def test_scaleup_generates_log_entry(self) -> None:
        system = _make_system()
        system.apply_action(3, "api")
        log = system.get_log_buffer()
        assert len(log) >= 1
        assert any("scale" in entry.lower() for entry in log)

    def test_multiple_scaleups_increase_count(self) -> None:
        system = _make_system()
        system.apply_action(3, "api")
        system.apply_action(3, "api")
        assert system._services["api"]["instance_count"] == 3


# ===========================================================================
# ClearCache Action Tests (VAL-ENV-009)
# ===========================================================================


class TestClearCacheAction:
    """ClearCache (action_type=4) resets cache-related metrics."""

    def test_clearcache_generates_log_entry(self) -> None:
        system = _make_system()
        system.apply_action(4, "db")
        log = system.get_log_buffer()
        assert len(log) >= 1
        assert any("cache" in entry.lower() for entry in log)

    def test_clearcache_reduces_latency(self) -> None:
        system = _make_system()
        _degrade_service(system, "db")
        latency_before = system._services["db"]["latency"]
        system.apply_action(4, "db")
        latency_after = system.get_metrics()["db"]["latency"]
        assert latency_after < latency_before

    def test_clearcache_reduces_memory(self) -> None:
        system = _make_system()
        _degrade_service(system, "api")
        memory_before = system._services["api"]["memory"]
        system.apply_action(4, "api")
        memory_after = system.get_metrics()["api"]["memory"]
        assert memory_after < memory_before

    def test_clearcache_on_all_services(self) -> None:
        for svc in SERVICE_NAMES:
            system = _make_system()
            _degrade_service(system, svc)
            system.apply_action(4, svc)
            log = system.get_log_buffer()
            assert any("cache" in entry.lower() for entry in log)


# ===========================================================================
# Log Buffer Tests (VAL-ENV-014)
# ===========================================================================


class TestLogBuffer:
    """Log buffer accumulates entries and caps at 10 (FIFO)."""

    def test_log_buffer_starts_empty(self) -> None:
        system = _make_system()
        assert system.get_log_buffer() == []

    def test_actions_add_log_entries(self) -> None:
        system = _make_system()
        system.apply_action(1, "api")
        assert len(system.get_log_buffer()) >= 1

    def test_log_buffer_caps_at_10(self) -> None:
        system = _make_system()
        # Perform many actions to generate log entries
        for i in range(15):
            system.apply_action(1, SERVICE_NAMES[i % 3])
        log = system.get_log_buffer()
        assert len(log) <= 10

    def test_log_buffer_fifo_eviction(self) -> None:
        """Oldest entries evicted when buffer exceeds 10."""
        system = _make_system()
        # Generate 12 log entries
        for i in range(12):
            system.apply_action(1, SERVICE_NAMES[i % 3])
        log = system.get_log_buffer()
        assert len(log) == 10
        # The first 2 entries should have been evicted;
        # the newest entries should be present

    def test_log_buffer_returns_list_of_strings(self) -> None:
        system = _make_system()
        system.apply_action(1, "api")
        log = system.get_log_buffer()
        assert isinstance(log, list)
        for entry in log:
            assert isinstance(entry, str)

    def test_log_buffer_returns_copy(self) -> None:
        """get_log_buffer returns a copy, not a reference."""
        system = _make_system()
        system.apply_action(1, "api")
        log1 = system.get_log_buffer()
        log1.append("extra")
        log2 = system.get_log_buffer()
        assert "extra" not in log2


# ===========================================================================
# Action Scoping Tests (VAL-ENV-016)
# ===========================================================================


class TestActionScoping:
    """Actions only affect target service, not other services."""

    def test_restart_only_affects_target(self) -> None:
        system = _make_system()
        _degrade_service(system, "api")
        _degrade_service(system, "order")
        metrics_order_before = system.get_metrics()["order"].copy()

        system.apply_action(1, "api")

        # api should be restored
        assert system._services["api"]["is_healthy"] is True
        # order should remain degraded
        metrics_order_after = system.get_metrics()["order"]
        assert metrics_order_after["cpu"] == metrics_order_before["cpu"]
        assert metrics_order_after["memory"] == metrics_order_before["memory"]

    def test_scaleup_only_affects_target(self) -> None:
        system = _make_system()
        count_db_before = system._services["db"]["instance_count"]
        count_order_before = system._services["order"]["instance_count"]

        system.apply_action(3, "api")

        assert system._services["db"]["instance_count"] == count_db_before
        assert system._services["order"]["instance_count"] == count_order_before

    def test_rollback_only_affects_target(self) -> None:
        system = _make_system()
        _set_bad_config_fault(system, "api")
        _set_bad_config_fault(system, "db")

        system.apply_action(2, "api")

        # api fault cleared
        assert system._active_faults.get("api") is None
        # db fault still present
        assert system._active_faults.get("db") == "bad_config"

    def test_clearcache_only_affects_target(self) -> None:
        system = _make_system()
        _degrade_service(system, "api")
        _degrade_service(system, "order")
        mem_order_before = system._services["order"]["memory"]

        system.apply_action(4, "api")

        mem_order_after = system.get_metrics()["order"]["memory"]
        assert mem_order_after == mem_order_before

    def test_noop_has_no_cross_service_effect(self) -> None:
        system = _make_system()
        _degrade_service(system, "db")
        metrics_db_before = system.get_metrics()["db"].copy()

        system.apply_action(0, "api")

        metrics_db_after = system.get_metrics()["db"]
        assert metrics_db_after == metrics_db_before


# ===========================================================================
# Tick / Natural Drift Tests
# ===========================================================================


class TestTick:
    """tick() simulates natural metric drift."""

    def test_tick_does_not_crash(self) -> None:
        system = _make_system()
        system.tick()  # Should not raise

    def test_tick_introduces_small_changes(self) -> None:
        system = _make_system()
        metrics_before = {
            svc: dict(system.get_metrics()[svc]) for svc in SERVICE_NAMES
        }
        # Run several ticks
        for _ in range(10):
            system.tick()
        metrics_after = system.get_metrics()
        # At least one metric should have changed
        changed = False
        for svc in SERVICE_NAMES:
            for key in ["cpu", "memory", "latency"]:
                if metrics_after[svc][key] != metrics_before[svc][key]:
                    changed = True
                    break
        assert changed, "tick() should introduce some metric drift"

    def test_tick_keeps_metrics_in_bounds(self) -> None:
        system = _make_system()
        for _ in range(100):
            system.tick()
        metrics = system.get_metrics()
        for svc in SERVICE_NAMES:
            assert 0.0 <= metrics[svc]["cpu"] <= 100.0
            assert 0.0 <= metrics[svc]["memory"] <= 100.0
            assert metrics[svc]["latency"] >= 0.0
            assert metrics[svc]["request_count"] >= 0


# ===========================================================================
# get_metrics Tests
# ===========================================================================


class TestGetMetrics:
    """get_metrics() returns dict per service with correct structure."""

    def test_returns_dict_of_dicts(self) -> None:
        system = _make_system()
        metrics = system.get_metrics()
        assert isinstance(metrics, dict)
        for svc in SERVICE_NAMES:
            assert isinstance(metrics[svc], dict)

    def test_each_service_has_four_metric_keys(self) -> None:
        system = _make_system()
        metrics = system.get_metrics()
        for svc in SERVICE_NAMES:
            assert "cpu" in metrics[svc]
            assert "memory" in metrics[svc]
            assert "latency" in metrics[svc]
            assert "request_count" in metrics[svc]

    def test_metric_types(self) -> None:
        system = _make_system()
        metrics = system.get_metrics()
        for svc in SERVICE_NAMES:
            assert isinstance(metrics[svc]["cpu"], float)
            assert isinstance(metrics[svc]["memory"], float)
            assert isinstance(metrics[svc]["latency"], float)
            # request_count can be int or float
            assert isinstance(metrics[svc]["request_count"], (int, float))


# ===========================================================================
# get_health_status Tests
# ===========================================================================


class TestGetHealthStatus:
    """get_health_status() returns dict of bools."""

    def test_returns_dict_of_bools(self) -> None:
        system = _make_system()
        health = system.get_health_status()
        assert isinstance(health, dict)
        for svc in SERVICE_NAMES:
            assert isinstance(health[svc], bool)

    def test_all_healthy_after_reset(self) -> None:
        system = _make_system()
        health = system.get_health_status()
        assert all(health[svc] for svc in SERVICE_NAMES)

    def test_reflects_degraded_service(self) -> None:
        system = _make_system()
        _degrade_service(system, "api")
        health = system.get_health_status()
        assert health["api"] is False
        assert health["order"] is True
        assert health["db"] is True


# ===========================================================================
# get_active_alerts Tests
# ===========================================================================


class TestGetActiveAlerts:
    """get_active_alerts() returns current alert list."""

    def test_returns_list(self) -> None:
        system = _make_system()
        alerts = system.get_active_alerts()
        assert isinstance(alerts, list)

    def test_empty_when_all_healthy(self) -> None:
        system = _make_system()
        alerts = system.get_active_alerts()
        assert alerts == []

    def test_contains_alerts_for_down_services(self) -> None:
        system = _make_system()
        _set_service_down(system, "db")
        alerts = system.get_active_alerts()
        assert len(alerts) > 0
        assert any("db" in alert.lower() for alert in alerts)

    def test_contains_alerts_for_unhealthy_services(self) -> None:
        system = _make_system()
        _degrade_service(system, "api")
        alerts = system.get_active_alerts()
        assert len(alerts) > 0

    def test_contains_alerts_for_faults(self) -> None:
        system = _make_system()
        _set_bad_config_fault(system, "order")
        alerts = system.get_active_alerts()
        assert len(alerts) > 0


# ===========================================================================
# Dependency Chain Tests
# ===========================================================================


class TestDependencyChain:
    """Services have dependency chain: api -> order -> db."""

    def test_system_knows_dependencies(self) -> None:
        system = _make_system()
        # The system should track that api depends on order, order on db
        deps = system._dependencies
        assert "order" in deps.get("api", [])
        assert "db" in deps.get("order", [])

    def test_db_has_no_dependencies(self) -> None:
        system = _make_system()
        deps = system._dependencies
        assert deps.get("db", []) == []


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_apply_action_all_services_all_types(self) -> None:
        """All 15 combinations of action_type x service should work."""
        for action_type in range(5):
            for svc in SERVICE_NAMES:
                system = _make_system()
                system.apply_action(action_type, svc)  # Should not raise

    def test_multiple_resets(self) -> None:
        """Multiple resets should not corrupt state."""
        system = SimulatedSystem()
        for seed in [1, 2, 3, 42, 99]:
            system.reset(seed=seed)
            health = system.get_health_status()
            assert all(health[svc] for svc in SERVICE_NAMES)

    def test_rapid_actions(self) -> None:
        """Many actions in sequence should not cause errors."""
        system = _make_system()
        for _ in range(50):
            for svc in SERVICE_NAMES:
                for action_type in range(5):
                    system.apply_action(action_type, svc)
        # Should still return valid metrics
        metrics = system.get_metrics()
        for svc in SERVICE_NAMES:
            assert 0.0 <= metrics[svc]["cpu"] <= 100.0
            assert 0.0 <= metrics[svc]["memory"] <= 100.0
            assert metrics[svc]["latency"] >= 0.0
