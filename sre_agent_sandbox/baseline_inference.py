"""LLM-based baseline inference script for the SRE Agent Sandbox.

Uses the OpenAI API client to run a language model against all 3 tasks
and produces reproducible graded scores (0.0-1.0).

Reads credentials from environment variables:
    OPENAI_API_KEY  — required
    OPENAI_MODEL    — optional, defaults to "gpt-4o-mini"

Usage::

    export OPENAI_API_KEY="sk-..."
    uv run python -m sre_agent_sandbox.baseline_inference
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from sre_agent_sandbox.models import SREAction, SREObservation, SREState
from sre_agent_sandbox.tasks import (
    TASKS,
    _configure_env_for_task,
    grade,
    make_env,
)

SYSTEM_PROMPT = """\
You are an SRE agent managing a 3-tier microservices system (api, order, db).

Each step you receive an observation with:
- metrics: per-service cpu, memory, latency, request_count
- health_status: per-service boolean
- active_alerts: list of alert strings
- log_buffer: recent log entries

You must respond with a JSON action:
{"action_type": <int>, "target_service": "<string>"}

Action types:
  0 = NoOp (do nothing)
  1 = RestartService (resets target to healthy baseline, clears all faults)
  2 = Rollback (reverts config, clears bad_config fault only)
  3 = ScaleUp (adds instance, reduces CPU and latency)
  4 = ClearCache (resets memory and latency)

Target services: "api", "order", "db"

Strategy:
- If you see "bad_config" in alerts → Rollback (type=2) on that service
- If you see "memory_leak" or high memory → RestartService (type=1) or ClearCache (type=4)
- If you see "latent_dependency" or high latency → RestartService (type=1) on the root cause
- If all services are healthy → NoOp (type=0)
- NEVER restart a healthy service (costs -10 penalty)

Respond with ONLY the JSON object, no explanation."""


def _format_observation(obs: SREObservation, state: SREState) -> str:
    """Format observation + state into a concise prompt for the LLM."""
    parts = []
    parts.append("=== Current Observation ===")
    parts.append(f"Step: {state.step_count}")
    parts.append(f"Health Score: {state.system_health_score:.2f}")

    parts.append("\nMetrics:")
    for svc in ["api", "order", "db"]:
        m = obs.metrics[svc]
        h = obs.health_status[svc]
        status = "UP" if h else "DOWN/DEGRADED"
        parts.append(
            f"  {svc}: status={status}, cpu={m['cpu']:.1f}%, "
            f"mem={m['memory']:.1f}%, lat={m['latency']:.0f}ms"
        )

    if obs.active_alerts:
        parts.append("\nActive Alerts:")
        for alert in obs.active_alerts:
            parts.append(f"  - {alert}")
    else:
        parts.append("\nNo active alerts.")

    if state.active_incidents:
        parts.append("\nActive Incidents:")
        for inc in state.active_incidents:
            parts.append(f"  - {inc}")

    return "\n".join(parts)


def _parse_action(response_text: str) -> SREAction:
    """Parse the LLM's response into an SREAction, with fallback."""
    text = response_text.strip()
    # Extract JSON from potential markdown code blocks
    if "```" in text:
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines).strip()

    try:
        data = json.loads(text)
        return SREAction(
            action_type=data["action_type"],
            target_service=data["target_service"],
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return SREAction(action_type=0, target_service="api")


class LLMAgent:
    """Agent that uses the OpenAI API to select actions."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError:
            print(
                "Error: openai package not installed. Run: uv pip install openai",
                file=sys.stderr,
            )
            sys.exit(1)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Error: OPENAI_API_KEY environment variable not set.",
                file=sys.stderr,
            )
            sys.exit(1)

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._messages: List[Dict[str, str]] = []

    def reset(self) -> None:
        """Clear conversation history for a new episode."""
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def act(self, obs: SREObservation, state: SREState) -> SREAction:
        """Query the LLM for an action given the current observation."""
        user_msg = _format_observation(obs, state)
        self._messages.append({"role": "user", "content": user_msg})

        # Keep conversation manageable (system + last 10 exchanges)
        if len(self._messages) > 21:
            self._messages = [self._messages[0]] + self._messages[-20:]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            temperature=0.0,
            max_tokens=100,
        )

        assistant_text = response.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": assistant_text})

        return _parse_action(assistant_text)


def evaluate_llm_agent(
    agent: LLMAgent,
    task_ids: List[str] | None = None,
) -> Dict[str, Any]:
    """Run the LLM agent on specified tasks and return graded results."""
    tasks_to_run = (
        [TASKS[tid] for tid in task_ids]
        if task_ids
        else [TASKS["sre_single_fault"], TASKS["sre_mixed_faults"], TASKS["sre_high_chaos"]]
    )

    results: Dict[str, Any] = {}

    for task in tasks_to_run:
        rewards: List[float] = []
        scores: List[float] = []

        for seed in task.eval_seeds:
            agent.reset()
            env = make_env(task, seed=seed)
            obs = env.reset(seed=seed)
            _configure_env_for_task(env, task)

            done = False
            while not done:
                action = agent.act(obs, env.state)
                obs = env.step(action)
                done = obs.done

            cum_reward = env._cumulative_reward
            rewards.append(cum_reward)
            scores.append(grade(task, cum_reward))

        mean_reward = sum(rewards) / len(rewards)
        mean_score = sum(scores) / len(scores)

        results[task.task_id] = {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "scores": scores,
            "mean_score": mean_score,
            "rewards": rewards,
            "mean_reward": mean_reward,
        }

    return results


def main() -> None:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    print("=" * 80)
    print("  SRE Agent Sandbox — LLM Baseline Inference")
    print(f"  Model: {model}")
    print("=" * 80)

    agent = LLMAgent(model=model)
    results = evaluate_llm_agent(agent)

    for task_id in ["sre_single_fault", "sre_mixed_faults", "sre_high_chaos"]:
        r = results[task_id]
        task = TASKS[task_id]
        print(f"\n{'─' * 80}")
        print(f"  Task: {task.name} ({task.difficulty})")
        print(f"{'─' * 80}")
        print(f"    Episodes:    {task.n_eval_episodes}")
        print(f"    Rewards:     {['%.2f' % x for x in r['rewards']]}")
        print(f"    Mean Reward: {r['mean_reward']:.3f}")
        print(f"    Scores:      {['%.3f' % s for s in r['scores']]}")
        print(f"    Mean Score:  {r['mean_score']:.3f}")

    print(f"\n{'=' * 80}")
    print("  Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
