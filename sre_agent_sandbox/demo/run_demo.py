"""Demo client for the SRE Agent Sandbox.

Runs N episodes with a random agent and N episodes with a heuristic agent,
printing the ASCII dashboard each step and episode summaries. At the end,
prints a comparison of mean cumulative rewards.

Usage::

    uv run python -m demo.run_demo          # top-level module invocation
    uv run python -m sre_agent_sandbox.demo.run_demo  # full package path

The heuristic agent reads active_incidents/alerts from the observation and
state to determine fault type:
  - Rollback for bad-config faults
  - RestartService for other faults (memory_leak, latent_dependency)
  - ScaleUp when services are overloaded (high CPU, no incidents)
  - NoOp when all services are healthy
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from sre_agent_sandbox.models import SREAction, SREObservation, SREState
from sre_agent_sandbox.server.environment import SREEnvironment
from sre_agent_sandbox.simulated_system import SERVICE_NAMES

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

SERVICE_LIST: List[str] = list(SERVICE_NAMES)


class RandomAgent:
    """Agent that selects random valid actions.

    Parameters
    ----------
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: SREObservation, state: SREState) -> SREAction:
        """Select a random action."""
        action_type = self._rng.randint(0, 4)
        target = self._rng.choice(SERVICE_LIST)
        return SREAction(action_type=action_type, target_service=target)


class HeuristicAgent:
    """Agent that uses a simple heuristic policy.

    Decision logic (evaluated in priority order):
      1. If any active incident contains 'bad_config' → Rollback (type=2) on
         the affected service.
      2. If any active incident contains another fault type → RestartService
         (type=1) on the affected service.
      3. If any service has CPU > 80% and no incidents → ScaleUp (type=3) on
         the highest-CPU service.
      4. Otherwise → NoOp (type=0).
    """

    def act(self, obs: SREObservation, state: SREState) -> SREAction:
        """Select an action based on the heuristic policy."""
        # Priority 1 & 2: Check active incidents from state
        incidents = state.active_incidents
        if incidents:
            return self._handle_incidents(incidents)

        # Priority 3: Check for overloaded services (high CPU)
        overloaded = self._find_overloaded_service(obs)
        if overloaded is not None:
            return SREAction(action_type=3, target_service=overloaded)

        # Priority 4: All healthy, do nothing
        return SREAction(action_type=0, target_service="api")

    def _handle_incidents(self, incidents: List[str]) -> SREAction:
        """Select an action to address the most critical incident."""
        # Check for bad_config first (highest priority for rollback)
        for incident in incidents:
            if "bad_config" in incident.lower():
                target = self._extract_target(incident)
                return SREAction(action_type=2, target_service=target)

        # Other faults: use RestartService
        for incident in incidents:
            target = self._extract_target(incident)
            return SREAction(action_type=1, target_service=target)

        # Fallback
        return SREAction(action_type=0, target_service="api")

    @staticmethod
    def _extract_target(incident: str) -> str:
        """Extract target service name from an incident string.

        Incident format examples:
          - 'bad_config on db'
          - 'memory_leak on order'
          - 'latent_dependency on api'
        """
        for svc in SERVICE_LIST:
            if svc in incident:
                return svc
        # Fallback to first service
        return "api"

    @staticmethod
    def _find_overloaded_service(obs: SREObservation) -> Optional[str]:
        """Return the service with CPU > 80%, or None."""
        max_cpu = 0.0
        max_svc: Optional[str] = None
        for svc in SERVICE_LIST:
            cpu = obs.metrics[svc]["cpu"]
            if cpu > 80.0 and cpu > max_cpu:
                max_cpu = cpu
                max_svc = svc
        return max_svc


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env: SREEnvironment,
    agent: RandomAgent | HeuristicAgent,
    seed: int = 42,
    render: bool = True,
) -> Dict[str, Any]:
    """Run a single episode and return a summary dict.

    Parameters
    ----------
    env:
        The SRE environment instance.
    agent:
        The agent to use for action selection.
    seed:
        Random seed for the environment.
    render:
        If True, print the ASCII dashboard each step.

    Returns
    -------
    dict
        Episode summary with keys: steps, reward, reason.
    """
    obs = env.reset(seed=seed)

    if render:
        _print_dashboard(env)

    done = False
    while not done:
        action = agent.act(obs, env.state)
        obs = env.step(action)
        done = obs.done

        if render:
            _print_dashboard(env)

    # Determine termination reason
    if env._step_count >= env._max_steps:
        reason = "max_steps"
    else:
        reason = "meltdown"

    return {
        "steps": env._step_count,
        "reward": env._cumulative_reward,
        "reason": reason,
    }


def _print_dashboard(env: SREEnvironment) -> None:
    """Print the ASCII dashboard for the current environment state."""
    # Import here to avoid circular import with the render parameter shadow
    from sre_agent_sandbox.renderer import render as render_fn

    print(render_fn(env))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    n_episodes: int = 5,
    max_steps: int = 200,
    render_steps: bool = True,
) -> None:
    """Run the full demo: random agent vs heuristic agent.

    Parameters
    ----------
    n_episodes:
        Number of episodes per agent.
    max_steps:
        Maximum steps per episode.
    render_steps:
        If True, print ASCII dashboard each step.
    """
    print("=" * 80)
    print("  SRE Agent Sandbox — Demo")
    print("=" * 80)

    # --- Random Agent ---
    print("\n" + "=" * 80)
    print("  Random Agent — {} episodes".format(n_episodes))
    print("=" * 80)

    random_results: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        print("\n--- Episode {}/{} ---".format(ep + 1, n_episodes))
        env = SREEnvironment(max_steps=max_steps, fault_probability=0.3)
        agent = RandomAgent(seed=ep)
        result = run_episode(env, agent, seed=ep, render=render_steps)
        random_results.append(result)
        print(
            "  Episode Summary: Steps={}, Reward={:.3f}, Reason={}".format(
                result["steps"], result["reward"], result["reason"]
            )
        )

    # --- Heuristic Agent ---
    print("\n" + "=" * 80)
    print("  Heuristic Agent — {} episodes".format(n_episodes))
    print("=" * 80)

    heuristic_results: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        print("\n--- Episode {}/{} ---".format(ep + 1, n_episodes))
        env = SREEnvironment(max_steps=max_steps, fault_probability=0.3)
        agent = HeuristicAgent()
        result = run_episode(env, agent, seed=ep, render=render_steps)
        heuristic_results.append(result)
        print(
            "  Episode Summary: Steps={}, Reward={:.3f}, Reason={}".format(
                result["steps"], result["reward"], result["reason"]
            )
        )

    # --- Comparison ---
    mean_random = sum(r["reward"] for r in random_results) / len(random_results)
    mean_heuristic = sum(r["reward"] for r in heuristic_results) / len(
        heuristic_results
    )

    print("\n" + "=" * 80)
    print("  Final Comparison")
    print("=" * 80)
    print("  Random Agent    — Mean Reward: {:.3f}".format(mean_random))
    print("  Heuristic Agent — Mean Reward: {:.3f}".format(mean_heuristic))
    if mean_heuristic > mean_random:
        print("  >> Heuristic outperforms Random by {:.3f}".format(
            mean_heuristic - mean_random
        ))
    else:
        print("  >> Random outperforms Heuristic by {:.3f}".format(
            mean_random - mean_heuristic
        ))
    print("=" * 80)


if __name__ == "__main__":
    main()
