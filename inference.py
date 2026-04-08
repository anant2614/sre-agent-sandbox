"""
Inference Script — SRE Agent Sandbox
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The Docker image name (only if using from_docker_image()).
    BASE_URL       The base URL for the SRE environment server.

- Defaults are set for API_BASE_URL, MODEL_NAME, and BASE_URL:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    BASE_URL     = os.getenv("BASE_URL", "https://iamanant-sre-agent-sandbox.hf.space")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=sre_mixed_faults env=sre-agent-sandbox model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=NoOp(api) reward=-0.50 done=false error=null
    [STEP] step=2 action=RestartService(db) reward=-10.10 done=false error=null
    [STEP] step=3 action=Rollback(order) reward=0.90 done=false error=null
    [END] success=true steps=3 score=0.650 rewards=-0.50,-10.10,0.90
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import SREEnv
from models import SREAction, SREObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("IMAGE_NAME")  # Docker image (if using from_docker_image)
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BASE_URL = os.getenv("BASE_URL") or "https://iamanant-sre-agent-sandbox.hf.space"

BENCHMARK = "sre-agent-sandbox"
TEMPERATURE = 0.0
MAX_TOKENS = 256

# Action names for readable logging
ACTION_NAMES: Dict[int, str] = {
    0: "NoOp",
    1: "RestartService",
    2: "Rollback",
    3: "ScaleUp",
    4: "ClearCache",
}

VALID_SERVICES = {"api", "order", "db"}

# ---------------------------------------------------------------------------
# Task definitions (mirrored from tasks.py for score normalisation)
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sre_single_fault": {
        "name": "Single Fault Remediation",
        "difficulty": "easy",
        "max_steps": 100,
        "reward_range": (-250.0, 100.0),
    },
    "sre_mixed_faults": {
        "name": "Mixed Fault Diagnosis",
        "difficulty": "medium",
        "max_steps": 200,
        "reward_range": (-1100.0, 200.0),
    },
    "sre_high_chaos": {
        "name": "High Chaos Survival",
        "difficulty": "hard",
        "max_steps": 300,
        "reward_range": (-1500.0, 300.0),
    },
}

# Which task(s) to run — set SRE_TASK env var or default to all three
SELECTED_TASK = os.getenv("SRE_TASK")  # e.g. "sre_mixed_faults" or None for all

# ---------------------------------------------------------------------------
# Grading (matches tasks.py:grade)
# ---------------------------------------------------------------------------


def grade(cumulative_reward: float, reward_range: tuple) -> float:
    """Normalise cumulative_reward to a score strictly within (0, 1)."""
    EPS = 0.001
    worst, best = reward_range
    if best == worst:
        raw = 1.0 if cumulative_reward >= best else 0.0
    else:
        raw = (cumulative_reward - worst) / (best - worst)
    return max(EPS, min(1.0 - EPS, raw))


# ---------------------------------------------------------------------------
# Logging helpers (exact format from sample-script.py)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------


def format_observation(obs: SREObservation) -> str:
    """Format an SRE observation into a concise summary for the LLM."""
    lines: List[str] = []

    # Health status
    health_parts = []
    for svc in ("api", "order", "db"):
        status = "HEALTHY" if obs.health_status.get(svc, False) else "UNHEALTHY"
        health_parts.append(f"{svc}={status}")
    lines.append(f"Health: {', '.join(health_parts)}")

    # Key metrics per service
    for svc in ("api", "order", "db"):
        m = obs.metrics.get(svc, {})
        lines.append(
            f"  {svc}: cpu={m.get('cpu', 0):.1f}% mem={m.get('memory', 0):.1f}% "
            f"latency={m.get('latency', 0):.0f}ms reqs={m.get('request_count', 0):.0f}"
        )

    # Active alerts
    if obs.active_alerts:
        lines.append(f"Alerts: {'; '.join(obs.active_alerts)}")
    else:
        lines.append("Alerts: none")

    # Recent logs (last 5 to keep prompt compact)
    if obs.log_buffer:
        recent = obs.log_buffer[-5:]
        lines.append("Recent logs:")
        for log_entry in recent:
            lines.append(f"  - {log_entry}")
    else:
        lines.append("Logs: none")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Site Reliability Engineer (SRE) agent managing a distributed \
    3-tier microservices system (api, order, db).

    ENVIRONMENT:
    - Each step you observe: service health, metrics (cpu, memory, latency, request_count), \
    active alerts, and recent logs.
    - You must choose ONE action and ONE target service each step.

    AVAILABLE ACTIONS:
    0 = NoOp        — Do nothing. Use when all services are healthy and metrics are normal.
    1 = RestartService — Restart a service. Fixes memory leaks but penalised (-10) if target is healthy.
    2 = Rollback     — Roll back a service config. Fixes bad_config faults.
    3 = ScaleUp      — Scale up a service. Can help with high load/latency.
    4 = ClearCache   — Clear a service cache. Helps reduce memory pressure.

    TARGET SERVICES: api, order, db

    FAULT TYPES TO DIAGNOSE:
    - memory_leak: Memory steadily increases. Fix with RestartService(target) or ClearCache(target).
    - latent_dependency: Latency increases on a service and cascades upstream (db→order→api). \
    Fix with ScaleUp(root_cause_service) or RestartService(root_cause_service).
    - bad_config: Service immediately becomes unhealthy. Fix with Rollback(target).

    REWARD STRUCTURE:
    - +1.0 per step when ALL services are healthy
    - Penalties for: high latency, service downtime (-5.0), unnecessary actions (-0.1), \
    restarting healthy services (-10.0)
    - Goal: Keep services healthy while minimising unnecessary interventions.

    STRATEGY:
    1. If all services are healthy and metrics normal → NoOp on any service.
    2. If a service is UNHEALTHY and you see "BadConfig" in alerts/logs → Rollback that service.
    3. If memory is rising (>70%) on a service → RestartService or ClearCache on that service.
    4. If latency is high (>200ms) → identify the root cause service (often db) and ScaleUp or RestartService it.
    5. Never restart a healthy service — it incurs a -10.0 penalty.

    RESPONSE FORMAT:
    Respond with ONLY a JSON object, nothing else:
    {"action_type": <0-4>, "target_service": "<api|order|db>", "reasoning": "<brief explanation>"}
""").strip()


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def build_user_prompt(
    step: int,
    obs: SREObservation,
    last_reward: float,
    history: List[str],
) -> str:
    """Build the user prompt with current observation and recent history."""
    obs_summary = format_observation(obs)
    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(f"""\
        Step: {step}
        Last reward: {last_reward:.2f}

        Current observation:
        {obs_summary}

        Recent history:
        {history_block}

        Choose your action (respond with JSON only).
    """).strip()


def parse_llm_response(response_text: str) -> SREAction:
    """Parse LLM JSON response into an SREAction. Falls back to NoOp on failure."""
    try:
        # Try to extract JSON from the response (handle markdown code blocks)
        text = response_text.strip()
        if text.startswith("```"):
            # Strip markdown fences
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        data = json.loads(text)
        action_type = int(data.get("action_type", 0))
        target_service = str(data.get("target_service", "api")).lower()

        # Validate
        if action_type not in ACTION_NAMES:
            action_type = 0
        if target_service not in VALID_SERVICES:
            target_service = "api"

        return SREAction(action_type=action_type, target_service=target_service)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"[DEBUG] Failed to parse LLM response: {exc}", flush=True)
        print(f"[DEBUG] Raw response: {response_text[:200]}", flush=True)
        return SREAction(action_type=0, target_service="api")


def get_model_action(
    client: Any,
    model_name: str,
    step: int,
    obs: SREObservation,
    last_reward: float,
    history: List[str],
) -> SREAction:
    """Query the LLM and return an SREAction."""
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    request_payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "stream": False,
    }
    try:
        # Most models use max_tokens; some newer endpoints require
        # max_completion_tokens instead.
        completion = client.chat.completions.create(
            **request_payload,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(text)
    except Exception as exc:
        if "max_completion_tokens" in str(exc):
            try:
                completion = client.chat.completions.create(
                    **request_payload,
                    max_completion_tokens=MAX_TOKENS,
                )
                text = (completion.choices[0].message.content or "").strip()
                return parse_llm_response(text)
            except Exception as retry_exc:
                print(f"[DEBUG] Model request failed: {retry_exc}", flush=True)
                return SREAction(action_type=0, target_service="api")

        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return SREAction(action_type=0, target_service="api")


def action_to_str(action: SREAction) -> str:
    """Format an SREAction as a readable string like 'RestartService(db)'."""
    name = ACTION_NAMES.get(action.action_type, "Unknown")
    return f"{name}({action.target_service})"


def build_llm_client_and_model() -> tuple[Any, str]:
    """Build OpenAI-compatible client and resolve model name from env."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return client, MODEL_NAME


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(
    llm_client: Any,
    model_name: str,
    env: SREEnv,
    task_id: str,
    task_config: Dict[str, Any],
) -> None:
    """Run a single episode for the given task."""
    max_steps = task_config["max_steps"]
    reward_range = tuple(task_config["reward_range"])

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    try:
        result = await env.reset()
        obs = result.observation
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action = get_model_action(
                llm_client,
                model_name,
                step,
                obs,
                last_reward,
                history,
            )
            action_str = action_to_str(action)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward={reward:+.2f} "
                f"health=[{','.join(k + '=' + ('OK' if v else 'BAD') for k, v in obs.health_status.items())}]"
            )

            if done:
                break

        cumulative = sum(rewards)
        score = grade(cumulative, reward_range)
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    llm_client, model_name = build_llm_client_and_model()

    # Determine which tasks to run
    if SELECTED_TASK:
        task_ids = [SELECTED_TASK]
    else:
        task_ids = list(TASK_CONFIGS.keys())

    for task_id in task_ids:
        if task_id not in TASK_CONFIGS:
            print(f"[DEBUG] Unknown task: {task_id}, skipping.", flush=True)
            continue

        task_config = TASK_CONFIGS[task_id]
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id} ({task_config['difficulty']})", flush=True)
        print(f"Max steps: {task_config['max_steps']}", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Connect to environment — use Docker image if provided, otherwise base_url
        if IMAGE_NAME:
            env = await SREEnv.from_docker_image(IMAGE_NAME)
        else:
            env = SREEnv(base_url=BASE_URL)

        try:
            async with env:
                await run_episode(llm_client, model_name, env, task_id, task_config)
        except Exception as exc:
            print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
