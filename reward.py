"""Reward computation for the SRE Agent Sandbox environment.

Computes a 5-component reward signal:
  1. **Availability**: +1.0 if ALL services healthy, 0.0 otherwise.
  2. **Latency penalty**: -0.01 * (avg_latency - 100) when avg > 100ms, else 0.0.
  3. **Downtime penalty**: -5.0 if ANY service is_down (once per step).
  4. **Efficiency penalty**: -0.1 for any non-NoOp action (action_type != 0).
  5. **Safety penalty**: -10.0 if RestartService (action_type == 1) on a healthy target.

The :meth:`calculate` method returns ``(total_reward, component_breakdown)``.
"""

from __future__ import annotations

from typing import Dict, Tuple

from models import SREAction
from simulated_system import SERVICE_NAMES, SimulatedSystem


class RewardCalculator:
    """Stateless reward calculator for the SRE environment.

    Call :meth:`calculate` each step to obtain the scalar reward and a
    per-component breakdown dictionary.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        system: SimulatedSystem,
        action: SREAction,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the reward for the current step.

        Parameters
        ----------
        system:
            The simulated system *after* the action has been applied and
            chaos has been ticked (i.e. the current observable state).
        action:
            The action taken by the agent this step.

        Returns
        -------
        tuple[float, dict[str, float]]
            ``(total_reward, breakdown)`` where *breakdown* maps component
            names to their individual contributions.
        """
        availability = self._availability(system)
        latency = self._latency_penalty(system)
        downtime = self._downtime_penalty(system)
        efficiency = self._efficiency_penalty(action)
        safety = self._safety_penalty(system, action)

        breakdown: Dict[str, float] = {
            "availability": availability,
            "latency": latency,
            "downtime": downtime,
            "efficiency": efficiency,
            "safety": safety,
        }

        total = availability + latency + downtime + efficiency + safety
        return total, breakdown

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _availability(system: SimulatedSystem) -> float:
        """+1.0 when **all** services healthy, 0.0 otherwise."""
        health = system.get_health_status()
        if all(health[svc] for svc in SERVICE_NAMES):
            return 1.0
        return 0.0

    @staticmethod
    def _latency_penalty(system: SimulatedSystem) -> float:
        """-0.01 * (avg_latency - 100) when avg > 100ms, else 0.0."""
        metrics = system.get_metrics()
        avg_latency = sum(metrics[svc]["latency"] for svc in SERVICE_NAMES) / len(SERVICE_NAMES)
        if avg_latency > 100.0:
            return -0.01 * (avg_latency - 100.0)
        return 0.0

    @staticmethod
    def _downtime_penalty(system: SimulatedSystem) -> float:
        """-5.0 if any service is_down (flat, once per step)."""
        for svc in SERVICE_NAMES:
            if system._services[svc]["is_down"]:
                return -5.0
        return 0.0

    @staticmethod
    def _efficiency_penalty(action: SREAction) -> float:
        """-0.1 for any non-NoOp action."""
        if action.action_type != 0:
            return -0.1
        return 0.0

    @staticmethod
    def _safety_penalty(system: SimulatedSystem, action: SREAction) -> float:
        """-10.0 for restarting a healthy service."""
        if action.action_type == 1:
            target_healthy = system._services[action.target_service]["is_healthy"]
            if target_healthy:
                return -10.0
        return 0.0
