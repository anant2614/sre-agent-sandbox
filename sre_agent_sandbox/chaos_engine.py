"""ChaosEngine: probabilistic fault injection for the SRE Agent Sandbox.

Supports three fault types:
  - **MemoryLeak**: gradual memory increase per tick until crash at 95%.
  - **LatentDependency**: progressive latency increase with upstream cascade
    through the dependency chain (db -> order -> api).
  - **BadConfig**: immediate service unhealthy with high error rate on
    injection step.

Uses a seeded ``random.Random(seed)`` for deterministic behaviour.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from sre_agent_sandbox.simulated_system import DEPENDENCIES, SERVICE_NAMES, SimulatedSystem

# ---------------------------------------------------------------------------
# Fault type constants
# ---------------------------------------------------------------------------
FAULT_TYPES: List[str] = ["memory_leak", "latent_dependency", "bad_config"]

# Memory leak parameters
MEMORY_LEAK_RATE: float = 8.0  # memory increase per tick
MEMORY_CRASH_THRESHOLD: float = 95.0  # service goes down at this memory %

# Latent dependency parameters
LATENCY_INCREMENT: float = 20.0  # latency increase per tick on target
LATENCY_CASCADE_FACTOR: float = 0.6  # fraction of latency increase passed upstream

# Reverse dependency map: service -> list of services that depend on it
# e.g. db is depended on by order, order is depended on by api
_REVERSE_DEPS: Dict[str, List[str]] = {}
for _svc, _deps in DEPENDENCIES.items():
    for _dep in _deps:
        _REVERSE_DEPS.setdefault(_dep, []).append(_svc)


class ChaosEngine:
    """Probabilistic fault injection engine.

    Parameters
    ----------
    fault_probability:
        Probability (0.0–1.0) that ``inject_fault`` injects a new fault
        each time it is called.
    seed:
        Random seed for deterministic fault sequences.
    """

    def __init__(self, fault_probability: float = 0.5, seed: int = 42) -> None:
        self._fault_probability = fault_probability
        self._rng = random.Random(seed)
        # Active faults: list of dicts with fault_type, target_service, and
        # any fault-specific state (e.g. accumulated memory for memory_leak).
        self._active_faults: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject_fault(self, system: SimulatedSystem) -> None:
        """Potentially inject a new fault based on *fault_probability*.

        A random roll determines whether injection occurs.  When it does,
        a random fault type is chosen and applied to a random service that
        does not already have an active fault of the same type.
        """
        if self._rng.random() >= self._fault_probability:
            return

        # Pick a random fault type
        fault_type = self._rng.choice(FAULT_TYPES)

        # Pick a target service that doesn't already have this fault type
        available = [
            s for s in SERVICE_NAMES
            if not any(
                f["fault_type"] == fault_type and f["target_service"] == s
                for f in self._active_faults
            )
        ]
        if not available:
            return

        target = self._rng.choice(available)
        self._inject_specific_fault(system, fault_type, target)

    def tick(self, system: SimulatedSystem) -> None:
        """Progress all active faults one step.

        - **MemoryLeak**: increases target memory by ``MEMORY_LEAK_RATE``;
          crashes service (is_down) when memory >= 95%.
        - **LatentDependency**: increases target latency by
          ``LATENCY_INCREMENT`` and cascades upstream.
        - **BadConfig**: no progressive effect (already applied on injection).
        """
        # Iterate over a copy since we may remove crashed-service faults
        for fault in list(self._active_faults):
            ft = fault["fault_type"]
            target = fault["target_service"]
            svc = system._services[target]

            if ft == "memory_leak":
                self._tick_memory_leak(system, fault, svc, target)
            elif ft == "latent_dependency":
                self._tick_latent_dependency(system, fault, svc, target)
            # bad_config: no tick progression needed

    def get_active_faults(self) -> List[Dict[str, Any]]:
        """Return a list of active fault descriptors (copies).

        Each descriptor contains at minimum ``fault_type`` and
        ``target_service``.
        """
        return [
            {"fault_type": f["fault_type"], "target_service": f["target_service"]}
            for f in self._active_faults
        ]

    def clear_all(self) -> None:
        """Remove all active faults."""
        self._active_faults.clear()

    # ------------------------------------------------------------------
    # Internal: specific fault injection
    # ------------------------------------------------------------------

    def _inject_specific_fault(
        self,
        system: SimulatedSystem,
        fault_type: str,
        target_service: str,
    ) -> None:
        """Inject a specific fault type on a specific service.

        This is also used by tests to inject known faults without
        probability gating.
        """
        fault: Dict[str, Any] = {
            "fault_type": fault_type,
            "target_service": target_service,
        }

        if fault_type == "memory_leak":
            # Record the starting memory so we can track progress
            fault["accumulated"] = 0.0
            self._active_faults.append(fault)

        elif fault_type == "latent_dependency":
            fault["accumulated_latency"] = 0.0
            self._active_faults.append(fault)

        elif fault_type == "bad_config":
            # Immediate effect: service goes unhealthy
            svc = system._services[target_service]
            svc["is_healthy"] = False
            system._active_faults[target_service] = "bad_config"
            system._add_log(
                f"BadConfig injected on {target_service}: service immediately unhealthy"
            )
            self._active_faults.append(fault)

    # ------------------------------------------------------------------
    # Internal: tick helpers
    # ------------------------------------------------------------------

    def _tick_memory_leak(
        self,
        system: SimulatedSystem,
        fault: Dict[str, Any],
        svc: Dict[str, Any],
        target: str,
    ) -> None:
        """Progress a memory leak fault by one step."""
        if svc["is_down"]:
            return

        svc["memory"] = min(100.0, svc["memory"] + MEMORY_LEAK_RATE)
        fault["accumulated"] += MEMORY_LEAK_RATE

        if svc["memory"] >= MEMORY_CRASH_THRESHOLD:
            svc["is_down"] = True
            svc["is_healthy"] = False
            system._add_log(
                f"MemoryLeak on {target}: memory hit {svc['memory']:.1f}%, service DOWN"
            )

    def _tick_latent_dependency(
        self,
        system: SimulatedSystem,
        fault: Dict[str, Any],
        svc: Dict[str, Any],
        target: str,
    ) -> None:
        """Progress a latent dependency fault and cascade upstream."""
        if svc["is_down"]:
            return

        # Increase latency on the directly targeted service
        svc["latency"] += LATENCY_INCREMENT
        fault["accumulated_latency"] += LATENCY_INCREMENT

        # Cascade upstream through the dependency chain
        self._cascade_latency(system, target, LATENCY_INCREMENT)

    def _cascade_latency(
        self,
        system: SimulatedSystem,
        source: str,
        latency_increase: float,
    ) -> None:
        """Propagate latency increase upstream from *source*.

        For each service that depends on *source*, add a fraction of the
        latency increase, then recurse upward.
        """
        upstream_services = _REVERSE_DEPS.get(source, [])
        for upstream in upstream_services:
            upstream_svc = system._services[upstream]
            if upstream_svc["is_down"]:
                continue
            cascade_amount = latency_increase * LATENCY_CASCADE_FACTOR
            upstream_svc["latency"] += cascade_amount
            # Continue cascading further upstream
            self._cascade_latency(system, upstream, cascade_amount)
