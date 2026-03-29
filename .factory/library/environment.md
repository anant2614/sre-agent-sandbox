# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Dependencies
- openenv-core>=0.2.2
- fastapi>=0.104.0
- uvicorn>=0.24.0
- pydantic>=2.0
- pytest, httpx, websockets (dev)

## Setup
- Python 3.10+ (developing on 3.13.7)
- uv for package management
- Docker for containerization
- No external services required — all simulation is in-memory
