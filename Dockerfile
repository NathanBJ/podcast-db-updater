FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src

ENV UV_HTTP_TIMEOUT=600
ENV PATH="/app/.venv/bin:$PATH"

RUN uv sync --no-install-project

CMD ["uv", "run", "/app/src/pipeline_update_db.py"]