"""Task definitions and graders for the SRE Agent Sandbox.

Defines three difficulty levels (easy, medium, hard), each with distinct
environment parameters and a grader that maps cumulative reward to a
normalised 0.0-1.0 score.

Task catalogue:
  - **sre_single_fault** (easy): only bad_config faults, low chaos.
  - **sre_mixed_faults** (medium): all fault types, moderate chaos.
  - **sre_high_chaos** (hard): all fault types, aggressive chaos with
    tighter latency thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from server.environment import SREEnvironment


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a single task (difficulty level).

    Parameters
    ----------
    task_id:
        Unique identifier for the task.
    name:
        Human-readable name.
    difficulty:
        One of "easy", "medium", "hard".
    description:
        Short description of the task scenario.
    max_steps:
        Episode step limit.
    fault_probability:
        Probability of a new fault each step.
    allowed_fault_types:
        Which fault types the chaos engine may inject.  ``None`` means all.
    n_eval_episodes:
        Number of episodes to average over for grading.
    eval_seeds:
        Deterministic seeds for reproducible evaluation.
    reward_range:
        ``(worst, best)`` cumulative-reward bounds used to normalise the
        grader score.  *worst* is derived from random-agent performance,
        *best* from a perfect theoretical agent.
    chaos_overrides:
        Optional dict of extra chaos-engine parameters (e.g. tighter
        latency threshold).
    """

    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    fault_probability: float
    allowed_fault_types: Optional[List[str]]
    n_eval_episodes: int
    eval_seeds: tuple[int, ...]
    reward_range: tuple[float, float]
    chaos_overrides: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

TASK_EASY = TaskConfig(
    task_id="sre_single_fault",
    name="Single Fault Remediation",
    difficulty="easy",
    description=(
        "Only bad_config faults are injected at a low rate.  The agent "
        "must learn to detect bad_config alerts and rollback the affected "
        "service.  A correct rollback policy solves this task."
    ),
    max_steps=100,
    fault_probability=0.1,
    allowed_fault_types=["bad_config"],
    n_eval_episodes=5,
    eval_seeds=(100, 101, 102, 103, 104),
    # Random ~-120, heuristic ~+88, perfect ~+100
    reward_range=(-250.0, 100.0),
)

TASK_MEDIUM = TaskConfig(
    task_id="sre_mixed_faults",
    name="Mixed Fault Diagnosis",
    difficulty="medium",
    description=(
        "All three fault types (memory_leak, latent_dependency, bad_config) "
        "are injected at a moderate rate.  The agent must diagnose the "
        "fault type from observations and choose the correct remediation "
        "action for each."
    ),
    max_steps=200,
    fault_probability=0.3,
    allowed_fault_types=None,
    n_eval_episodes=5,
    eval_seeds=(200, 201, 202, 203, 204),
    # Random ~-840, heuristic ~-190, perfect ~+200
    reward_range=(-1100.0, 200.0),
)

TASK_HARD = TaskConfig(
    task_id="sre_high_chaos",
    name="High Chaos Survival",
    difficulty="hard",
    description=(
        "All fault types at aggressive injection rate with tighter "
        "latency timeout thresholds.  Multiple concurrent faults and "
        "cascading failures require rapid, precise triage to keep "
        "services alive."
    ),
    max_steps=300,
    fault_probability=0.5,
    allowed_fault_types=None,
    n_eval_episodes=5,
    eval_seeds=(300, 301, 302, 303, 304),
    # Random ~-480, heuristic ~-670, perfect ~+300
    reward_range=(-1500.0, 300.0),
    chaos_overrides={"latency_timeout_threshold": 350.0},
)

TASKS: Dict[str, TaskConfig] = {
    t.task_id: t for t in [TASK_EASY, TASK_MEDIUM, TASK_HARD]
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(task: TaskConfig, seed: Optional[int] = None) -> SREEnvironment:
    """Create an ``SREEnvironment`` configured for *task*.

    When the task restricts ``allowed_fault_types``, the chaos engine's
    ``inject_fault`` is monkey-patched to only inject from that subset.
    """
    env = SREEnvironment(
        max_steps=task.max_steps,
        fault_probability=task.fault_probability,
    )
    return env


def _configure_env_for_task(env: SREEnvironment, task: TaskConfig) -> None:
    """Apply task-specific configuration after reset.

    Must be called after ``env.reset()`` so that the chaos engine and
    system instances exist.
    """
    # Restrict fault types if specified
    if task.allowed_fault_types is not None:
        env._chaos._allowed_fault_types = list(task.allowed_fault_types)

    # Apply chaos overrides as instance-level parameters
    if "latency_timeout_threshold" in task.chaos_overrides:
        env._chaos._latency_timeout_threshold = task.chaos_overrides[
            "latency_timeout_threshold"
        ]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade(task: TaskConfig, cumulative_reward: float) -> float:
    """Normalise *cumulative_reward* to a score strictly within (0, 1).

    Uses linear interpolation between the task's worst-case and best-case
    reward bounds, clamped to (EPS, 1-EPS) so scores are never exactly
    0.0 or 1.0.
    """
    EPS = 1e-4
    worst, best = task.reward_range
    if best == worst:
        raw = 1.0 if cumulative_reward >= best else 0.0
    else:
        raw = (cumulative_reward - worst) / (best - worst)
    return max(EPS, min(1.0 - EPS, raw))


def evaluate_agent(
    task: TaskConfig,
    agent,
    render: bool = False,
) -> Dict[str, Any]:
    """Run the agent on all eval episodes for *task* and return graded results.

    Parameters
    ----------
    task:
        The task configuration.
    agent:
        Any object with an ``act(obs, state) -> SREAction`` method.
    render:
        If True, print ASCII dashboard each step.

    Returns
    -------
    dict
        Keys: task_id, difficulty, scores (list[float]), mean_score (float),
        rewards (list[float]), mean_reward (float).
    """
    rewards: List[float] = []
    scores: List[float] = []

    for seed in task.eval_seeds:
        env = make_env(task, seed=seed)
        obs = env.reset(seed=seed)
        _configure_env_for_task(env, task)

        if render:
            from renderer import render as render_fn
            print(render_fn(env))

        done = False
        while not done:
            action = agent.act(obs, env.state)
            obs = env.step(action)
            done = obs.done

            if render:
                from renderer import render as render_fn
                print(render_fn(env))

        cum_reward = env._cumulative_reward
        rewards.append(cum_reward)
        scores.append(grade(task, cum_reward))

    mean_reward = sum(rewards) / len(rewards)
    mean_score = sum(scores) / len(scores)

    return {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "scores": scores,
        "mean_score": mean_score,
        "rewards": rewards,
        "mean_reward": mean_reward,
    }
