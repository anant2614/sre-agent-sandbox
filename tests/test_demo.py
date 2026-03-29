"""Tests for the demo client script.

Tests cover:
  - Random agent completes full episodes without crashing
  - Heuristic agent completes full episodes without crashing
  - Heuristic outperforms random in mean cumulative reward
  - Heuristic selects rollback for bad-config, restart for other faults
  - Demo prints episode summary with steps, reward, reason
  - Demo prints final reward comparison
"""

from __future__ import annotations

import pytest

from sre_agent_sandbox.demo.run_demo import (
    HeuristicAgent,
    RandomAgent,
    main,
    run_episode,
)
from sre_agent_sandbox.models import SREAction
from sre_agent_sandbox.server.environment import SREEnvironment

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
