"""SREEnvironment — main OpenEnv environment for the SRE Agent Sandbox.

Ties together :class:`SimulatedSystem`, :class:`ChaosEngine`, and
:class:`RewardCalculator` into a fully compliant OpenEnv ``Environment``
subclass with ``reset()`` / ``step()`` / ``state`` API.

Termination conditions:
  - ``step_count >= max_steps`` (default 200)
  - Total meltdown: all 3 services are down simultaneously
"""

from __future__ import annotations

import uuid
from typing import Optional

from openenv.core.env_server.interfaces import Environment

from sre_agent_sandbox.chaos_engine import ChaosEngine
from sre_agent_sandbox.models import SREAction, SREObservation, SREState
from sre_agent_sandbox.reward import RewardCalculator
from sre_agent_sandbox.simulated_system import SERVICE_NAMES, SimulatedSystem

# Default maximum steps per episode
DEFAULT_MAX_STEPS: int = 200


class SREEnvironment(Environment[SREAction, SREObservation, SREState]):
    """OpenEnv environment simulating a 3-tier microservices system under stress.

    Parameters
    ----------
    max_steps:
        Maximum number of steps per episode before termination.
    fault_probability:
        Probability of a new fault injection each step (0.0–1.0).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        fault_probability: float = 0.3,
    ) -> None:
        self._max_steps = max_steps
        self._fault_probability = fault_probability

        # Components (instantiated on reset)
        self._system: SimulatedSystem = SimulatedSystem()
        self._chaos: ChaosEngine = ChaosEngine(fault_probability=fault_probability)
        self._reward_calc: RewardCalculator = RewardCalculator()

        # Episode state
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._is_reset: bool = False
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SREObservation:
        """Reset the environment to a healthy baseline state.

        Parameters
        ----------
        seed:
            Random seed for deterministic episode replay.
        episode_id:
            Custom episode identifier.  Auto-generated if ``None``.

        Returns
        -------
        SREObservation
            Initial observation with all services healthy and done=False.
        """
        # Generate or accept episode ID
        self._episode_id = episode_id if episode_id is not None else str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._is_reset = True
        self._cumulative_reward = 0.0

        # Reset sub-components
        self._system = SimulatedSystem()
        self._system.reset(seed=seed)

        self._chaos = ChaosEngine(
            fault_probability=self._fault_probability,
            seed=seed if seed is not None else 42,
        )

        return self._build_observation(reward=None, done=False)

    # ------------------------------------------------------------------
    # OpenEnv API: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SREObservation:
        """Execute one step: apply action, inject chaos, compute reward.

        Parameters
        ----------
        action:
            The SRE agent's chosen action.
        timeout_s:
            Unused (no real I/O).

        Returns
        -------
        SREObservation
            The observation after the step, including reward and done flag.

        Raises
        ------
        RuntimeError
            If called before ``reset()`` or after the episode has ended.
        """
        if not self._is_reset:
            raise RuntimeError(
                "Environment has not been reset. Call reset() before step()."
            )
        if self._done:
            raise RuntimeError(
                "Episode has already ended (done=True). Call reset() to start a new episode."
            )

        # 1. Apply the agent's action to the simulated system
        self._system.apply_action(action.action_type, action.target_service)

        # 2. Chaos engine: potentially inject new faults and tick existing ones
        self._chaos.inject_fault(self._system)
        self._chaos.tick(self._system)

        # 3. Natural metric drift
        self._system.tick()

        # 4. Increment step count
        self._step_count += 1

        # 5. Compute reward
        reward, _ = self._reward_calc.calculate(self._system, action)
        self._cumulative_reward += reward

        # 6. Check termination conditions
        done = self._check_termination()
        self._done = done

        return self._build_observation(reward=reward, done=done)

    # ------------------------------------------------------------------
    # OpenEnv API: state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> SREState:
        """Return the current environment state."""
        health_status = self._system.get_health_status()
        healthy_count = sum(1 for h in health_status.values() if h)
        health_score = healthy_count / len(SERVICE_NAMES)

        # Active incidents: combine active faults from chaos engine and system
        active_incidents = []
        for fault in self._chaos.get_active_faults():
            active_incidents.append(
                f"{fault['fault_type']} on {fault['target_service']}"
            )
        # Also include system-level faults not tracked by chaos engine
        for svc, fault_type in self._system._active_faults.items():
            incident = f"{fault_type} on {svc}"
            if incident not in active_incidents:
                active_incidents.append(incident)

        return SREState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            system_health_score=health_score,
            active_incidents=active_incidents,
        )

    # ------------------------------------------------------------------
    # Render (ASCII dashboard)
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Produce an ASCII dashboard of the current environment state.

        Returns
        -------
        str
            Multi-line ASCII string suitable for terminal display.
            All lines are <= 120 characters, using only printable ASCII.
        """
        metrics = self._system.get_metrics()
        health = self._system.get_health_status()
        alerts = self._system.get_active_alerts()
        st = self.state

        lines: list[str] = []
        sep = "=" * 80

        lines.append(sep)
        lines.append(f"  SRE Agent Sandbox  |  Episode: {st.episode_id or 'N/A'}")
        lines.append(f"  Step: {st.step_count:<6}  |  Health Score: {st.system_health_score:.2f}")
        lines.append(f"  Cumulative Reward: {self._cumulative_reward:.3f}")
        lines.append(sep)

        # Service table header
        lines.append(
            f"  {'Service':<10} {'Status':<12} {'CPU%':<8} {'Mem%':<8} "
            f"{'Lat(ms)':<10} {'Req/s':<8}"
        )
        lines.append("-" * 70)

        for svc in SERVICE_NAMES:
            m = metrics[svc]
            h = health[svc]
            is_down = self._system._services[svc]["is_down"]
            if is_down:
                status = "DOWN"
            elif h:
                status = "UP"
            else:
                status = "DEGRADED"

            lines.append(
                f"  {svc:<10} {status:<12} {m['cpu']:>6.1f}  {m['memory']:>6.1f}  "
                f"{m['latency']:>8.1f}  {m['request_count']:>6.0f}"
            )

        lines.append("-" * 70)

        # Active alerts
        lines.append("  Active Alerts:")
        if alerts:
            for alert in alerts:
                lines.append(f"    - {alert}")
        else:
            lines.append("    (none)")

        # Active incidents
        if st.active_incidents:
            lines.append("  Active Incidents:")
            for inc in st.active_incidents:
                lines.append(f"    - {inc}")

        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float | None,
        done: bool,
    ) -> SREObservation:
        """Construct an SREObservation from the current system state."""
        return SREObservation(
            metrics=self._system.get_metrics(),
            log_buffer=self._system.get_log_buffer(),
            health_status=self._system.get_health_status(),
            active_alerts=self._system.get_active_alerts(),
            done=done,
            reward=reward,
        )

    def _check_termination(self) -> bool:
        """Return True if the episode should end."""
        # Max steps reached
        if self._step_count >= self._max_steps:
            return True

        # Total meltdown: all services down
        all_down = all(
            self._system._services[svc]["is_down"] for svc in SERVICE_NAMES
        )
        if all_down:
            return True

        return False
