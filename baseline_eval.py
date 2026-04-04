"""Baseline evaluation script for the SRE Agent Sandbox.

Runs random and heuristic agents across all three tasks (easy, medium, hard)
and reports reproducible graded scores (0.0-1.0).

Usage::

    uv run python -m baseline_eval
"""

from __future__ import annotations

from demo.run_demo import HeuristicAgent, RandomAgent
from tasks import TASKS, evaluate_agent


def main() -> None:
    agents = {
        "Random": RandomAgent(seed=0),
        "Heuristic": HeuristicAgent(),
    }

    print("=" * 80)
    print("  SRE Agent Sandbox — Baseline Evaluation")
    print("=" * 80)

    tasks_ordered = [TASKS["sre_single_fault"], TASKS["sre_mixed_faults"], TASKS["sre_high_chaos"]]

    for task in tasks_ordered:
        print(f"\n{'─' * 80}")
        print(f"  Task: {task.name} ({task.difficulty})")
        print(f"  {task.description}")
        print(f"{'─' * 80}")

        for agent_name, agent in agents.items():
            result = evaluate_agent(task, agent, render=False)
            print(f"\n  {agent_name} Agent:")
            print(f"    Episodes:    {task.n_eval_episodes}")
            print(f"    Rewards:     {['%.2f' % r for r in result['rewards']]}")
            print(f"    Mean Reward: {result['mean_reward']:.3f}")
            print(f"    Scores:      {['%.3f' % s for s in result['scores']]}")
            print(f"    Mean Score:  {result['mean_score']:.3f}")

    print(f"\n{'=' * 80}")
    print("  Done. Scores are normalised to 0.0-1.0 (higher is better).")
    print("=" * 80)


if __name__ == "__main__":
    main()
