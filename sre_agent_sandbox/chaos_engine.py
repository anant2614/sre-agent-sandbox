"""ChaosEngine: probabilistic fault injection for the SRE Agent Sandbox.

Supports three fault types:
  - **MemoryLeak**: gradual memory increase per tick until crash at 95%.
  - **LatentDependency**: progressive latency increase with upstream cascade
    through the dependency chain (db -> order -> api).  When latency exceeds
    the timeout threshold, timeout alerts/logs are emitted for the service
    and its upstream callers.
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
LATENCY_TIMEOUT_THRESHOLD: float = 500.0  # latency above this triggers timeout alerts

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

    def remove_faults_for_service(
        self,
        target_service: str,
        system: SimulatedSystem | None = None,
    ) -> None:
        """Remove all active faults targeting *target_service*.

        Used when a remediation action successfully resolves a fault,
        so that the chaos engine's fault tracking stays in sync with
        the simulated system state.

        For ``latent_dependency`` faults, if *system* is provided, the
        cascaded latency contributions on upstream services are also
        removed so that upstream latencies normalise after the root
        cause is remediated.
        """
        remaining: List[Dict[str, Any]] = []
        for f in self._active_faults:
            if f["target_service"] == target_service:
                # If this is a latent_dependency fault, undo cascade contributions
                if f["fault_type"] == "latent_dependency" and system is not None:
                    cascade_contribs: Dict[str, float] = f.get(
                        "cascade_contributions", {}
                    )
                    for svc_name, amount in cascade_contribs.items():
                        svc = system._services[svc_name]
                        svc["latency"] = max(0.0, svc["latency"] - amount)

                    # Remove timeout alerts that were added by this fault
                    system._active_alerts = [
                        a for a in system._active_alerts
                        if not (
                            a.startswith("Timeout:")
                            and (
                                target_service in a
                                or f"cascade from {target_service}" in a
                            )
                        )
                    ]
                # Drop this fault (don't append to remaining)
            else:
                remaining.append(f)
        self._active_faults = remaining

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
            # Track cumulative cascaded latency per upstream service so we
            # can undo it when the fault is remediated.
            fault["cascade_contributions"] = {}  # type: Dict[str, float]
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
        """Progress a latent dependency fault and cascade upstream.

        When any service's latency exceeds ``LATENCY_TIMEOUT_THRESHOLD``,
        timeout alerts and log entries are emitted for the service and
        its upstream callers.
        """
        if svc["is_down"]:
            return

        # Increase latency on the directly targeted service
        svc["latency"] += LATENCY_INCREMENT
        fault["accumulated_latency"] += LATENCY_INCREMENT

        # Cascade upstream through the dependency chain, tracking contributions
        self._cascade_latency(system, target, LATENCY_INCREMENT, fault)

        # --- Timeout signalling (VAL-CHAOS-002) ---
        self._check_timeout_alerts(system, target, fault)

    def _cascade_latency(
        self,
        system: SimulatedSystem,
        source: str,
        latency_increase: float,
        fault: Dict[str, Any] | None = None,
    ) -> None:
        """Propagate latency increase upstream from *source*.

        For each service that depends on *source*, add a fraction of the
        latency increase, then recurse upward.  When *fault* is provided,
        the cascaded amounts are recorded in ``fault["cascade_contributions"]``
        so they can be undone when the fault is remediated.
        """
        upstream_services = _REVERSE_DEPS.get(source, [])
        for upstream in upstream_services:
            upstream_svc = system._services[upstream]
            if upstream_svc["is_down"]:
                continue
            cascade_amount = latency_increase * LATENCY_CASCADE_FACTOR
            upstream_svc["latency"] += cascade_amount

            # Track cumulative cascade contribution for recovery
            if fault is not None:
                contribs = fault.setdefault("cascade_contributions", {})
                contribs[upstream] = contribs.get(upstream, 0.0) + cascade_amount

            # Continue cascading further upstream
            self._cascade_latency(system, upstream, cascade_amount, fault)

    def _check_timeout_alerts(
        self,
        system: SimulatedSystem,
        target: str,
        fault: Dict[str, Any],
    ) -> None:
        """Emit timeout alerts/logs when latency exceeds the threshold.

        Checks the directly targeted service and all upstream services
        that have cascaded latency.  Alerts are only emitted once per
        service per fault (tracked via ``fault["timeout_alerted"]``).
        """
        alerted: set = fault.setdefault("timeout_alerted", set())

        # Check the directly targeted service
        svc = system._services[target]
        if svc["latency"] > LATENCY_TIMEOUT_THRESHOLD and target not in alerted:
            alert_msg = f"Timeout: {target} latency exceeded threshold"
            system._active_alerts.append(alert_msg)
            system._add_log(
                f"LatentDependency on {target}: latency "
                f"{svc['latency']:.0f}ms exceeded {LATENCY_TIMEOUT_THRESHOLD:.0f}ms threshold"
            )
            alerted.add(target)

        # Check upstream services that have cascaded latency
        self._check_upstream_timeout(system, target, fault, alerted)

    def _check_upstream_timeout(
        self,
        system: SimulatedSystem,
        source: str,
        fault: Dict[str, Any],
        alerted: set,
    ) -> None:
        """Recursively check upstream services for timeout and emit alerts."""
        upstream_services = _REVERSE_DEPS.get(source, [])
        for upstream in upstream_services:
            upstream_svc = system._services[upstream]
            if upstream_svc["is_down"]:
                continue
            if (
                upstream_svc["latency"] > LATENCY_TIMEOUT_THRESHOLD
                and upstream not in alerted
            ):
                alert_msg = (
                    f"Timeout: {upstream} latency exceeded threshold "
                    f"(cascade from {fault['target_service']})"
                )
                system._active_alerts.append(alert_msg)
                system._add_log(
                    f"Timeout cascade: {upstream} latency "
                    f"{upstream_svc['latency']:.0f}ms exceeded threshold "
                    f"(dependency on {fault['target_service']})"
                )
                alerted.add(upstream)
            # Continue checking further upstream
            self._check_upstream_timeout(system, upstream, fault, alerted)
