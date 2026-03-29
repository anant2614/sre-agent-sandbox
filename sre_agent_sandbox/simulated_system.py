"""Simulated 3-tier system state model for the SRE Agent Sandbox.

Maintains internal state for a 3-tier application:
  - API Gateway
  - Order Service
  - DB/Cache

Each service tracks: cpu, memory, latency, request_count, is_healthy,
is_down, and instance_count.  The class provides methods for reset,
action application, natural metric drift (tick), and state queries.

Dependency chain: api -> order -> db
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Baseline metric values for healthy services
# ---------------------------------------------------------------------------
BASELINE_METRICS: Dict[str, Any] = {
    "cpu": 30.0,
    "memory": 40.0,
    "latency": 50.0,
    "request_count": 100,
    "is_healthy": True,
    "is_down": False,
    "instance_count": 1,
}

SERVICE_NAMES: List[str] = ["api", "order", "db"]

# Dependency chain: service -> list of services it depends on
DEPENDENCIES: Dict[str, List[str]] = {
    "api": ["order"],
    "order": ["db"],
    "db": [],
}

# Maximum log buffer size (FIFO eviction)
MAX_LOG_BUFFER: int = 10


class SimulatedSystem:
    """Internal state model for a 3-tier microservices application.

    Provides reset, action application, natural drift, and query methods
    consumed by :class:`SREEnvironment` and :class:`ChaosEngine`.
    """

    def __init__(self) -> None:
        self._services: Dict[str, Dict[str, Any]] = {}
        self._log_buffer: List[str] = []
        self._active_alerts: List[str] = []
        self._active_faults: Dict[str, str] = {}
        self._dependencies: Dict[str, List[str]] = dict(DEPENDENCIES)
        self._rng: Optional[random.Random] = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """Restore all services to healthy baseline and clear all state.

        Parameters
        ----------
        seed:
            Optional random seed for deterministic metric drift via
            :pymethod:`tick`.
        """
        self._rng = random.Random(seed)
        self._log_buffer = []
        self._active_alerts = []
        self._active_faults = {}

        self._services = {}
        for name in SERVICE_NAMES:
            self._services[name] = dict(BASELINE_METRICS)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def apply_action(self, action_type: int, target_service: str) -> None:
        """Execute a remediation action on *target_service*.

        Action types:
            0 – NoOp (do nothing)
            1 – RestartService (reset metrics to baseline)
            2 – Rollback (revert to stable config, clear bad_config fault)
            3 – ScaleUp (increase instance_count, reduce latency/cpu)
            4 – ClearCache (reset cache-related metrics: latency, memory)
        """
        if action_type == 0:
            # NoOp – intentionally does nothing
            return

        if action_type == 1:
            self._action_restart(target_service)
        elif action_type == 2:
            self._action_rollback(target_service)
        elif action_type == 3:
            self._action_scaleup(target_service)
        elif action_type == 4:
            self._action_clearcache(target_service)

    # -- Individual action handlers ----------------------------------------

    def _action_restart(self, target: str) -> None:
        """RestartService: reset target to healthy baseline metrics."""
        svc = self._services[target]
        svc["cpu"] = BASELINE_METRICS["cpu"]
        svc["memory"] = BASELINE_METRICS["memory"]
        svc["latency"] = BASELINE_METRICS["latency"]
        svc["request_count"] = BASELINE_METRICS["request_count"]
        svc["is_healthy"] = True
        svc["is_down"] = False
        # instance_count is preserved (restart doesn't change scaling)
        self._add_log(f"RestartService applied to {target}: metrics reset to baseline")

    def _action_rollback(self, target: str) -> None:
        """Rollback: revert to stable config, clears bad_config fault."""
        svc = self._services[target]
        had_bad_config = self._active_faults.get(target) == "bad_config"

        # Clear bad_config fault if present
        if had_bad_config:
            del self._active_faults[target]
            # Restore metrics to baseline for the service
            svc["cpu"] = BASELINE_METRICS["cpu"]
            svc["memory"] = BASELINE_METRICS["memory"]
            svc["latency"] = BASELINE_METRICS["latency"]
            svc["request_count"] = BASELINE_METRICS["request_count"]
            svc["is_healthy"] = True
            svc["is_down"] = False

        self._add_log(
            f"Rollback applied to {target}: "
            f"{'bad_config cleared, metrics restored' if had_bad_config else 'stable config reverted'}"
        )

    def _action_scaleup(self, target: str) -> None:
        """ScaleUp: increase instance_count and reduce latency/cpu proportionally."""
        svc = self._services[target]
        old_count = svc["instance_count"]
        new_count = old_count + 1
        svc["instance_count"] = new_count

        # Reduce CPU and latency proportionally to the scaling factor
        scale_factor = old_count / new_count
        svc["cpu"] = max(0.0, svc["cpu"] * scale_factor)
        svc["latency"] = max(0.0, svc["latency"] * scale_factor)

        self._add_log(
            f"ScaleUp applied to {target}: instances {old_count} -> {new_count}"
        )

    def _action_clearcache(self, target: str) -> None:
        """ClearCache: reset cache-related metrics (latency and memory)."""
        svc = self._services[target]
        # Reset memory and latency toward baseline
        svc["memory"] = BASELINE_METRICS["memory"]
        svc["latency"] = BASELINE_METRICS["latency"]

        self._add_log(f"ClearCache applied to {target}: cache metrics reset")

    # ------------------------------------------------------------------
    # Tick (natural metric drift)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Simulate natural metric drift for all services.

        Small random fluctuations are applied to cpu, memory, latency, and
        request_count.  Metrics are clamped to valid ranges.
        """
        if self._rng is None:
            return

        for name in SERVICE_NAMES:
            svc = self._services[name]
            if svc["is_down"]:
                continue

            # Small random drift (±2 for cpu/memory, ±5 for latency, ±10 for requests)
            svc["cpu"] = max(0.0, min(100.0, svc["cpu"] + self._rng.uniform(-2.0, 2.0)))
            svc["memory"] = max(0.0, min(100.0, svc["memory"] + self._rng.uniform(-2.0, 2.0)))
            svc["latency"] = max(0.0, svc["latency"] + self._rng.uniform(-5.0, 5.0))
            svc["request_count"] = max(0, int(svc["request_count"] + self._rng.uniform(-10, 10)))

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return per-service metrics: ``{service: {cpu, memory, latency, request_count}}``."""
        result: Dict[str, Dict[str, float]] = {}
        for name in SERVICE_NAMES:
            svc = self._services[name]
            result[name] = {
                "cpu": float(svc["cpu"]),
                "memory": float(svc["memory"]),
                "latency": float(svc["latency"]),
                "request_count": float(svc["request_count"]),
            }
        return result

    def get_health_status(self) -> Dict[str, bool]:
        """Return per-service health status: ``{service: bool}``."""
        return {name: self._services[name]["is_healthy"] for name in SERVICE_NAMES}

    def get_log_buffer(self) -> List[str]:
        """Return the last 10 log entries (FIFO).  Returns a copy."""
        return list(self._log_buffer)

    def get_active_alerts(self) -> List[str]:
        """Return currently active alert messages.

        Alerts are dynamically generated based on service state:
        - Services that are down
        - Services that are unhealthy
        - Active faults
        """
        alerts: List[str] = []

        for name in SERVICE_NAMES:
            svc = self._services[name]
            if svc["is_down"]:
                alerts.append(f"{name} is DOWN")
            elif not svc["is_healthy"]:
                alerts.append(f"{name} is UNHEALTHY")

        for service, fault_type in self._active_faults.items():
            alerts.append(f"Active fault on {service}: {fault_type}")

        return alerts

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _add_log(self, message: str) -> None:
        """Append a log message, evicting the oldest if buffer exceeds cap."""
        self._log_buffer.append(message)
        if len(self._log_buffer) > MAX_LOG_BUFFER:
            self._log_buffer = self._log_buffer[-MAX_LOG_BUFFER:]
