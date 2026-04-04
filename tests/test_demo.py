"""Tests for the demo client script.

Tests cover:
  - Random agent completes valid random actions
  - Heuristic agent inspects obs.active_alerts for fault-type detection
  - Bad-config alerts trigger Rollback action
  - Memory-leak alerts trigger RestartService
  - Timeout/latency alerts trigger RestartService on affected dependency
  - High-CPU alerts trigger ScaleUp
  - No alerts → NoOp
  - Heuristic outperforms random in mean cumulative reward
  - Demo prints episode summary with steps, reward, reason
  - Demo prints final reward comparison
"""

from __future__ import annotations

import pytest

from demo.run_demo import (
    HeuristicAgent,
    RandomAgent,
    main,
    run_episode,
)
from models import SREAction, SREObservation, SREState
from server.environment import SREEnvironment

# ---------------------------------------------------------------------------
# Agent Tests
# ---------------------------------------------------------------------------


class TestRandomAgent:
    """RandomAgent selects valid random actions."""

    def test_returns_sre_action(self) -> None:
        agent = RandomAgent(seed=42)
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        action = agent.act(obs, env.state)
        assert isinstance(action, SREAction)

    def test_action_type_in_range(self) -> None:
        agent = RandomAgent(seed=42)
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        for _ in range(50):
            action = agent.act(obs, env.state)
            assert 0 <= action.action_type <= 4
            assert action.target_service in ("api", "order", "db")
            obs = env.step(action)
            if obs.done:
                break


class TestHeuristicAgent:
    """HeuristicAgent uses smart action selection."""

    def test_returns_sre_action(self) -> None:
        agent = HeuristicAgent()
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        action = agent.act(obs, env.state)
        assert isinstance(action, SREAction)

    def test_noop_when_healthy(self) -> None:
        """When all services are healthy, heuristic should select NoOp."""
        agent = HeuristicAgent()
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        action = agent.act(obs, env.state)
        assert action.action_type == 0, "Heuristic should NoOp when all services healthy"

    def test_rollback_for_bad_config(self) -> None:
        """Heuristic selects rollback (type=2) for bad-config faults."""
        agent = HeuristicAgent()
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        # Inject bad_config fault on db
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")
        # Re-read observation after fault injection
        obs = env._build_observation(reward=None, done=False)
        action = agent.act(obs, env.state)
        assert action.action_type == 2, "Heuristic should rollback for bad_config"
        assert action.target_service == "db"

    def test_restart_for_other_faults(self) -> None:
        """Heuristic selects restart (type=1) for non-bad-config faults."""
        agent = HeuristicAgent()
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        # Inject memory_leak fault on order
        env._chaos._inject_specific_fault(env._system, "memory_leak", "order")
        env._chaos.tick(env._system)  # progress to make it visible
        obs = env._build_observation(reward=None, done=False)
        action = agent.act(obs, env.state)
        assert action.action_type == 1, "Heuristic should restart for memory_leak"
        assert action.target_service == "order"

    def test_scaleup_when_overloaded(self) -> None:
        """Heuristic selects ScaleUp (type=3) when CPU is high but no incidents."""
        agent = HeuristicAgent()
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        # Manually set high CPU for api
        env._system._services["api"]["cpu"] = 90.0
        obs = env._build_observation(reward=None, done=False)
        action = agent.act(obs, env.state)
        assert action.action_type == 3, "Heuristic should ScaleUp when CPU is high"
        assert action.target_service == "api"


class TestHeuristicAlertParsing:
    """HeuristicAgent.act() inspects obs.active_alerts for fault-type detection."""

    @staticmethod
    def _make_obs_with_alerts(alerts: list[str]) -> SREObservation:
        """Create a minimal valid SREObservation with custom active_alerts."""
        return SREObservation(
            metrics={
                svc: {"cpu": 30.0, "memory": 40.0, "latency": 50.0, "request_count": 100}
                for svc in ("api", "order", "db")
            },
            log_buffer=[],
            health_status={"api": True, "order": True, "db": True},
            active_alerts=alerts,
            done=False,
            reward=None,
        )

    @staticmethod
    def _make_state(incidents: list[str] | None = None) -> SREState:
        """Create a minimal valid SREState."""
        return SREState(
            episode_id="test-ep",
            step_count=1,
            system_health_score=1.0,
            active_incidents=incidents or [],
        )

    def test_bad_config_alert_triggers_rollback(self) -> None:
        """Alert containing 'bad_config' → Rollback (type=2)."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(["Active fault on db: bad_config"])
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 2, "bad_config alert should trigger Rollback"
        assert action.target_service == "db"

    def test_memory_leak_alert_triggers_restart(self) -> None:
        """Alert containing 'memory_leak' → RestartService (type=1)."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(["Active fault on order: memory_leak"])
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 1, "memory_leak alert should trigger RestartService"
        assert action.target_service == "order"

    def test_high_memory_alert_triggers_restart(self) -> None:
        """Alert containing 'high_memory' → RestartService (type=1)."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(["high_memory on api"])
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 1, "high_memory alert should trigger RestartService"
        assert action.target_service == "api"

    def test_timeout_alert_triggers_restart(self) -> None:
        """Alert containing 'timeout' → RestartService (type=1)."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(
            ["Timeout: order latency exceeded threshold"]
        )
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 1, "Timeout alert should trigger RestartService"
        assert action.target_service == "order"

    def test_cascade_timeout_targets_root_cause(self) -> None:
        """Cascade timeout alert → RestartService on the cascade source."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(
            ["Timeout: api latency exceeded threshold (cascade from db)"]
        )
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 1
        assert action.target_service == "db", (
            "Cascade timeout should target root-cause service 'db'"
        )

    def test_latency_alert_triggers_restart(self) -> None:
        """Alert containing 'latency' → RestartService (type=1)."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(["db latency above threshold"])
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 1, "Latency alert should trigger RestartService"
        assert action.target_service == "db"

    def test_high_cpu_alert_triggers_scaleup(self) -> None:
        """Alert containing 'high_cpu' → ScaleUp (type=3)."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(["high_cpu on api"])
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 3, "high_cpu alert should trigger ScaleUp"
        assert action.target_service == "api"

    def test_latent_dependency_alert_triggers_restart(self) -> None:
        """Alert containing 'latent_dependency' → RestartService (type=1)."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(
            ["Active fault on db: latent_dependency"]
        )
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 1
        assert action.target_service == "db"

    def test_no_alerts_no_incidents_gives_noop(self) -> None:
        """No alerts and no incidents → NoOp (type=0)."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts([])
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 0, "No alerts should give NoOp"

    def test_alerts_take_priority_over_incidents(self) -> None:
        """When both alerts and incidents exist, alerts are inspected first."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(["Active fault on db: bad_config"])
        state = self._make_state(incidents=["memory_leak on order"])
        action = agent.act(obs, state)
        # Alert says bad_config on db → Rollback on db (not restart on order)
        assert action.action_type == 2
        assert action.target_service == "db"

    def test_bad_config_has_priority_over_other_alerts(self) -> None:
        """bad_config alert should be handled before other fault alerts."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts([
            "Active fault on order: memory_leak",
            "Active fault on db: bad_config",
        ])
        state = self._make_state()
        action = agent.act(obs, state)
        assert action.action_type == 2, "bad_config should take priority"
        assert action.target_service == "db"

    def test_status_only_alerts_fall_through_to_incidents(self) -> None:
        """Status alerts ('is DOWN') without fault info fall through to incidents."""
        agent = HeuristicAgent()
        obs = self._make_obs_with_alerts(["api is DOWN"])
        state = self._make_state(incidents=["memory_leak on api"])
        action = agent.act(obs, state)
        # The 'is DOWN' alert has no actionable fault info, so it falls
        # through to incidents where memory_leak triggers RestartService.
        assert action.action_type == 1
        assert action.target_service == "api"

    def test_integrated_env_bad_config_alert(self) -> None:
        """Integration: inject bad_config via chaos, heuristic reads obs.active_alerts."""
        agent = HeuristicAgent()
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        # Inject bad_config fault
        env._chaos._inject_specific_fault(env._system, "bad_config", "db")
        obs = env._build_observation(reward=None, done=False)
        # Verify the alert actually appears in obs.active_alerts
        assert any("bad_config" in a for a in obs.active_alerts), (
            f"Expected bad_config alert in {obs.active_alerts}"
        )
        action = agent.act(obs, env.state)
        assert action.action_type == 2, "Should rollback for bad_config alert"
        assert action.target_service == "db"

    def test_integrated_env_memory_leak_via_incidents(self) -> None:
        """Integration: inject memory_leak, heuristic falls through alerts to incidents."""
        agent = HeuristicAgent()
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        # Inject memory_leak fault (tracked in chaos engine, surfaces via incidents)
        env._chaos._inject_specific_fault(env._system, "memory_leak", "order")
        env._chaos.tick(env._system)
        obs = env._build_observation(reward=None, done=False)
        state = env.state
        # memory_leak should appear in state.active_incidents
        assert any("memory_leak" in inc for inc in state.active_incidents), (
            f"Expected memory_leak incident in {state.active_incidents}"
        )
        action = agent.act(obs, state)
        assert action.action_type == 1, "Should restart for memory_leak incident"
        assert action.target_service == "order"

    def test_integrated_env_memory_leak_alert_when_unhealthy(self) -> None:
        """Integration: tick memory_leak until service is UNHEALTHY, alert appears."""
        agent = HeuristicAgent()
        env = SREEnvironment(max_steps=50, fault_probability=0.0)
        obs = env.reset(seed=42)
        # Inject and tick memory_leak enough to make service unhealthy
        env._chaos._inject_specific_fault(env._system, "memory_leak", "order")
        for _ in range(10):
            env._chaos.tick(env._system)
        obs = env._build_observation(reward=None, done=False)
        # Service should be unhealthy or down, generating alerts
        has_alert = any(
            "order" in a.lower() for a in obs.active_alerts
        )
        assert has_alert, (
            f"Expected order-related alert in {obs.active_alerts}"
        )
        action = agent.act(obs, env.state)
        # Should pick restart for order (either from alerts or incidents)
        assert action.action_type == 1
        assert action.target_service == "order"


# ---------------------------------------------------------------------------
# Episode Tests
# ---------------------------------------------------------------------------


class TestRunEpisode:
    """run_episode completes without crashing."""

    def test_random_agent_completes_episode(self) -> None:
        env = SREEnvironment(max_steps=50, fault_probability=0.3)
        agent = RandomAgent(seed=42)
        result = run_episode(env, agent, seed=42, render=False)
        assert "steps" in result
        assert "reward" in result
        assert "reason" in result
        assert result["steps"] > 0
        assert isinstance(result["reward"], (int, float))

    def test_heuristic_agent_completes_episode(self) -> None:
        env = SREEnvironment(max_steps=50, fault_probability=0.3)
        agent = HeuristicAgent()
        result = run_episode(env, agent, seed=42, render=False)
        assert "steps" in result
        assert "reward" in result
        assert "reason" in result
        assert result["steps"] > 0

    def test_episode_terminates_at_max_steps(self) -> None:
        env = SREEnvironment(max_steps=10, fault_probability=0.0)
        agent = RandomAgent(seed=42)
        result = run_episode(env, agent, seed=42, render=False)
        assert result["steps"] == 10
        assert result["reason"] == "max_steps"

    def test_episode_result_has_reason(self) -> None:
        env = SREEnvironment(max_steps=50, fault_probability=0.3)
        agent = RandomAgent(seed=42)
        result = run_episode(env, agent, seed=42, render=False)
        assert result["reason"] in ("max_steps", "meltdown")


# ---------------------------------------------------------------------------
# Heuristic vs Random comparison
# ---------------------------------------------------------------------------


class TestHeuristicOutperformsRandom:
    """Over multiple episodes, heuristic should outperform random."""

    def test_heuristic_better_mean_reward(self) -> None:
        n_episodes = 5
        max_steps = 50

        random_rewards = []
        for i in range(n_episodes):
            env = SREEnvironment(max_steps=max_steps, fault_probability=0.3)
            agent = RandomAgent(seed=i)
            result = run_episode(env, agent, seed=i, render=False)
            random_rewards.append(result["reward"])

        heuristic_rewards = []
        for i in range(n_episodes):
            env = SREEnvironment(max_steps=max_steps, fault_probability=0.3)
            agent = HeuristicAgent()
            result = run_episode(env, agent, seed=i, render=False)
            heuristic_rewards.append(result["reward"])

        mean_random = sum(random_rewards) / len(random_rewards)
        mean_heuristic = sum(heuristic_rewards) / len(heuristic_rewards)
        assert mean_heuristic > mean_random, (
            f"Heuristic ({mean_heuristic:.2f}) should outperform "
            f"random ({mean_random:.2f})"
        )


# ---------------------------------------------------------------------------
# Main function test (captures stdout)
# ---------------------------------------------------------------------------


class TestMainFunction:
    """main() runs to completion and produces expected output."""

    def test_main_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(n_episodes=2, max_steps=20, render_steps=False)
        captured = capsys.readouterr()
        assert "Random Agent" in captured.out
        assert "Heuristic Agent" in captured.out

    def test_main_prints_episode_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(n_episodes=2, max_steps=20, render_steps=False)
        captured = capsys.readouterr()
        # Episode summary should contain steps, reward, and reason
        assert "steps" in captured.out.lower() or "Steps" in captured.out
        assert "reward" in captured.out.lower() or "Reward" in captured.out

    def test_main_prints_comparison(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(n_episodes=2, max_steps=20, render_steps=False)
        captured = capsys.readouterr()
        assert "comparison" in captured.out.lower() or "Comparison" in captured.out or "vs" in captured.out.lower()
