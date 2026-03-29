"""ASCII renderer for the SRE Agent Sandbox environment.

Produces a terminal dashboard showing:
  - Service health grid (service name + UP/DOWN/DEGRADED status)
  - Metrics table (CPU%, Memory%, Latency ms per service)
  - Active alerts section
  - Cumulative episode reward

All output lines are <= 120 characters wide using only printable ASCII.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sre_agent_sandbox.simulated_system import SERVICE_NAMES

if TYPE_CHECKING:
    from sre_agent_sandbox.server.environment import SREEnvironment


def render(env: SREEnvironment) -> str:
    """Produce an ASCII dashboard of the current environment state.

    Parameters
    ----------
    env:
        The SREEnvironment instance to render.

    Returns
    -------
    str
        Multi-line ASCII string suitable for terminal display.
        All lines are <= 120 characters, using only printable ASCII.
    """
    metrics = env._system.get_metrics()
    health = env._system.get_health_status()
    alerts = env._system.get_active_alerts()
    st = env.state

    lines: list[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append(
        "  SRE Agent Sandbox  |  Episode: {}".format(
            _truncate(st.episode_id or "N/A", 50)
        )
    )
    lines.append(
        "  Step: {:<6}  |  Health Score: {:.2f}".format(
            st.step_count, st.system_health_score
        )
    )
    lines.append("  Cumulative Reward: {:.3f}".format(env._cumulative_reward))
    lines.append(sep)

    # Service table header
    lines.append(
        "  {:<10} {:<12} {:<8} {:<8} {:<10} {:<8}".format(
            "Service", "Status", "CPU%", "Mem%", "Lat(ms)", "Req/s"
        )
    )
    lines.append("-" * 70)

    for svc in SERVICE_NAMES:
        m = metrics[svc]
        is_down = env._system._services[svc]["is_down"]
        h = health[svc]

        if is_down:
            status = "DOWN"
        elif h:
            status = "UP"
        else:
            status = "DEGRADED"

        lines.append(
            "  {:<10} {:<12} {:>6.1f}  {:>6.1f}  {:>8.1f}  {:>6.0f}".format(
                svc, status, m["cpu"], m["memory"], m["latency"], m["request_count"]
            )
        )

    lines.append("-" * 70)

    # Active alerts
    lines.append("  Active Alerts:")
    if alerts:
        for alert in alerts:
            lines.append("    - {}".format(_truncate(alert, 110)))
    else:
        lines.append("    (none)")

    # Active incidents
    if st.active_incidents:
        lines.append("  Active Incidents:")
        for inc in st.active_incidents:
            lines.append("    - {}".format(_truncate(inc, 110)))

    lines.append(sep)

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* characters, adding '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
