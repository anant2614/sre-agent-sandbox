#!/bin/bash
set -e

cd /Users/anant/ai/sre-agent-sandbox

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is required but not installed"
    exit 1
fi

# Create venv and install dependencies (idempotent)
if [ ! -d ".venv" ]; then
    uv venv
fi
uv sync

echo "Environment ready."
