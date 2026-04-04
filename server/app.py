"""FastAPI application for the SRE Agent Sandbox.

Uses OpenEnv's ``create_app`` to generate the standard HTTP server with all
required endpoints: ``/health``, ``/schema``, ``/metadata``, ``/reset``,
``/step``, ``/state``, ``/ws``, and ``/mcp``.

Usage::

    uv run server
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app

from models import SREAction, SREObservation
from server.environment import SREEnvironment

app = create_app(
    SREEnvironment,
    SREAction,
    SREObservation,
    env_name="sre_agent_sandbox",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
