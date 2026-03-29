"""Top-level entry point for ``python -m demo.run_demo``.

Delegates to :func:`sre_agent_sandbox.demo.run_demo.main`.
"""

from __future__ import annotations

from sre_agent_sandbox.demo.run_demo import main

if __name__ == "__main__":
    main()
