# 1. Start with the lightweight Python Slim image
FROM python:3.12-slim

# 2. The Magic Trick: Copy the 'uv' binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Verify it works
RUN uv --version

# 4. Create user early (adds ~1KB to image)
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m appuser

RUN mkdir /app
WORKDIR /app

COPY pyproject.toml ./
ENV UV_HTTP_TIMEOUT=600

# 5. Let uv create the .venv
RUN uv sync --no-install-project

# 6. Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 7. CRITICAL: Add the venv to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# 8. Copy application code
COPY src ./src

# 9. Change ownership and switch user (single layer)
RUN chown -R appuser:appgroup /app

USER appuser

CMD ["uv", "run", "/app/src/pipeline_update_db.py"]