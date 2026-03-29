"""Tests for the ASCII renderer module.

Tests cover:
  - render() shows service names with UP/DOWN/DEGRADED indicators
  - render() shows CPU, Memory, Latency metrics per service
  - render() shows active alerts (or None when healthy)
  - render() shows cumulative reward
  - render() output <= 120 chars per line, printable ASCII only
"""

from __future__ import annotations

import string

import pytest

from sre_agent_sandbox.models import SREAction
from sre_agent_sandbox.renderer import render
from sre_agent_sandbox.server.environment import SREEnvironment
from sre_agent_sandbox.simulated_system import SERVICE_NAMES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRINTABLE_ASCII = set(string.printable)


def _all_printable_ascii(text: str) -> bool:
    """Return True if *text* contains only printable ASCII characters."""
    return all(ch in PRINTABLE_ASCII for ch in text)


def _max_line_width(text: str) -> int:
    """Return the maximum line width in *text*."""
    return max((len(line) for line in text.split("\n")), default=0)


# ---------------------------------------------------------------------------
# Tests: Healthy state render
# ---------------------------------------------------------------------------


class TestRenderHealthy:
    """render() on a freshly reset (all-healthy) environment."""

    @pytest.fixture()
    def env(self) -> SREEnvironment:
        env = SREEnvironment(max_steps=200, fault_probability=0.0)
        env.reset(seed=42)
        return env

    def test_returns_string(self, env: SREEnvironment) -> None:
        output = render(env)
        assert isinstance(output, str)

    def test_contains_service_names(self, env: SREEnvironment) -> None:
        output = render(env)
        for svc in SERVICE_NAMES:
            assert svc in output, f"Service '{svc}' not found in render output"

    def test_contains_up_status_for_healthy(self, env: SREEnvironment) -> None:
        output = render(env)
        assert "UP" in output

    def test_contains_cpu_metric(self, env: SREEnvironment) -> None:
        output = render(env)
        assert "CPU" in output

    def test_contains_memory_metric(self, env: SREEnvironment) -> None:
        output = render(env)
        assert "Mem" in output

    def test_contains_latency_metric(self, env: SREEnvironment) -> None:
        output = render(env)
        assert "Lat" in output

    def test_contains_cumulative_reward(self, env: SREEnvironment) -> None:
        output = render(env)
        assert "Reward" in output

    def test_healthy_no_alert_content(self, env: SREEnvironment) -> None:
        """When all services healthy, alerts section says 'none' or similar."""
        output = render(env)
        # Should mention alerts section exists
        assert "Alert" in output
        # Should indicate none
        assert "none" in output.lower() or "None" in output

    def test_max_line_width(self, env: SREEnvironment) -> None:
        output = render(env)
        assert _max_line_width(output) <= 120

    def test_printable_ascii_only(self, env: SREEnvironment) -> None:
        output = render(env)
        assert _all_printable_ascii(output), "Output contains non-printable ASCII characters"


# ---------------------------------------------------------------------------
# Tests: Degraded / faulted state render
# ---------------------------------------------------------------------------


class TestRenderDegraded:
    """render() with faults injected to test DOWN/DEGRADED indicators."""

    @pytest.fixture()
    def env_with_fault(self) -> SREEnvironment:
        """Create an environment with a bad_config fault on db."""
        env = SREEnvironment(max_steps=200, fault_probability=0.0)
        env.reset(seed=42)
        # Manually inject a bad_config fault
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")
        return env

    def test_contains_degraded_or_down_indicator(self, env_with_fault: SREEnvironment) -> None:
        output = render(env_with_fault)
        # db should be DEGRADED (unhealthy but not is_down from bad_config)
        assert "DEGRADED" in output or "DOWN" in output

    def test_shows_active_alerts(self, env_with_fault: SREEnvironment) -> None:
        output = render(env_with_fault)
        # Should show an active alert for the fault
        assert "bad_config" in output.lower() or "fault" in output.lower() or "UNHEALTHY" in output

    def test_max_line_width_with_faults(self, env_with_fault: SREEnvironment) -> None:
        output = render(env_with_fault)
        assert _max_line_width(output) <= 120

    def test_printable_ascii_with_faults(self, env_with_fault: SREEnvironment) -> None:
        output = render(env_with_fault)
        assert _all_printable_ascii(output)


class TestRenderAfterSteps:
    """render() after taking some steps, verifying cumulative reward updates."""

    def test_cumulative_reward_nonzero_after_steps(self) -> None:
        env = SREEnvironment(max_steps=200, fault_probability=0.0)
        env.reset(seed=42)
        # Take a NoOp action (should give +1.0 availability)
        action = SREAction(action_type=0, target_service="api")
        env.step(action)
        output = render(env)
        # Should show a non-zero cumulative reward
        assert "Reward" in output
        # Cumulative reward after one NoOp on healthy system should be ~1.0
        assert "0.000" not in output or "1." in output

    def test_multiple_steps_changes_display(self) -> None:
        env = SREEnvironment(max_steps=200, fault_probability=0.0)
        env.reset(seed=42)
        action = SREAction(action_type=0, target_service="api")
        for _ in range(5):
            env.step(action)
        output = render(env)
        assert "Step" in output
        assert "5" in output  # step count should be 5

    def test_width_constraint_after_many_steps(self) -> None:
        env = SREEnvironment(max_steps=200, fault_probability=0.3)
        env.reset(seed=123)
        action = SREAction(action_type=0, target_service="api")
        for _ in range(20):
            obs = env.step(action)
            if obs.done:
                break
        output = render(env)
        assert _max_line_width(output) <= 120
        assert _all_printable_ascii(output)


class TestRenderDownService:
    """render() when a service is completely down."""

    def test_shows_down_status(self) -> None:
        env = SREEnvironment(max_steps=200, fault_probability=0.0)
        env.reset(seed=42)
        # Force db to be down
        env._system._services["db"]["is_down"] = True
        env._system._services["db"]["is_healthy"] = False
        output = render(env)
        assert "DOWN" in output
