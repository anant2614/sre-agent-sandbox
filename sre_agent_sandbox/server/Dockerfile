# NOTE: This is a copy of the root-level Dockerfile for reference.
# Build from the project root: docker build -t sre-agent-sandbox .

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

# Copy source code
COPY sre_agent_sandbox/ sre_agent_sandbox/
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
    rm -rf /app/.venv/lib/python3.12/site-packages/gradio/templates 2>/dev/null; \
    find /app/.venv/lib/python3.12/site-packages/gradio -type f \( -name "*.js" -o -name "*.css" -o -name "*.map" -o -name "*.woff*" -o -name "*.ttf" \) -delete 2>/dev/null; \
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
COPY --from=builder /app/sre_agent_sandbox /app/sre_agent_sandbox
COPY --from=builder /app/openenv.yaml /app/openenv.yaml
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Ensure the venv Python is on PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "sre_agent_sandbox.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
