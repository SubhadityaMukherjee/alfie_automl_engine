FROM python:3.12-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl ca-certificates git build-essential \
  libgl1 libglib2.0-0 tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

ENV PATH="/root/.local/bin/:$PATH"
COPY app/ ./app

WORKDIR /app
COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen

EXPOSE 8001

ENV HEALTHCHECK_PORT=8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${HEALTHCHECK_PORT}/health || exit 1
