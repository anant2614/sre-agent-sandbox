"""Top-level server entry point for OpenEnv multi-mode deployment.

Re-exports the FastAPI ``app`` from the main package and provides a
``main()`` function so that ``uv run server`` and ``python -m server.app``
both work.
"""

from sre_agent_sandbox.server.app import app  # noqa: F401


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
