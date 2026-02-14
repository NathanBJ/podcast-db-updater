# =============================================================================
# STAGE 1: Build stage (as root, for speed)
# =============================================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./
ENV UV_HTTP_TIMEOUT=600

# Install dependencies as root (faster)
RUN uv sync --no-install-project

# =============================================================================
# STAGE 2: Final image (as appuser)
# =============================================================================
FROM python:3.12-slim

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create user FIRST
#RUN groupadd -g 1000 appgroup && \
#    useradd -u 1000 -g appgroup -m appuser

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy .venv from builder WITH correct ownership (no extra layer!)
COPY --from=builder /app/.venv /app/.venv

# Copy source code with correct ownership
#COPY --chown=appuser:appgroup pyproject.toml uv.lock ./
#COPY --chown=appuser:appgroup src ./src
COPY pyproject.toml uv.lock ./
COPY src ./src

# Set PATH
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
#USER appuser

CMD ["uv", "run", "/app/src/pipeline_update_db.py"]