# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies with uv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the system Python (no venv needed in container)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code (flat package layout at repo root)
COPY models.py client.py simulated_system.py chaos_engine.py reward.py renderer.py tasks.py baseline_eval.py baseline_inference.py __init__.py ./
COPY server/ server/
COPY demo/ demo/
COPY openenv.yaml ./

# Install the project itself
RUN uv sync --frozen --no-dev

# Trim heavy transitive dependencies that are not needed at runtime.
# openenv-core pulls in gradio, pandas, numpy, pillow, etc. but the
# SRE-agent-sandbox only uses the core types (Action, Observation, State,
# Environment) plus FastAPI / uvicorn.  Removing unused artefacts (caches,
# frontend JS/CSS bundles, test suites) keeps the image under the 500 MB budget.
RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -type f -name "*.pyc" -delete 2>/dev/null; \
    find /app/.venv -type f -name "*.pyo" -delete 2>/dev/null; \
    rm -rf /app/.venv/lib/python3.12/site-packages/pandas/tests 2>/dev/null; \
    rm -rf /app/.venv/lib/python3.12/site-packages/numpy/tests 2>/dev/null; \
    rm -rf /app/.venv/lib/python3.12/site-packages/numpy/*/tests 2>/dev/null; \
    rm -rf /app/.venv/lib/python3.12/site-packages/PIL/tests 2>/dev/null; \
    true

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim image with only what's needed
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models.py /app/client.py /app/simulated_system.py /app/chaos_engine.py /app/reward.py /app/renderer.py /app/tasks.py /app/baseline_eval.py /app/baseline_inference.py /app/__init__.py /app/
COPY --from=builder /app/server /app/server
COPY --from=builder /app/demo /app/demo
COPY --from=builder /app/openenv.yaml /app/openenv.yaml
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Ensure the venv Python is on PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
