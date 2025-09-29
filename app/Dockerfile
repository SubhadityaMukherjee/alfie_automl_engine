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
COPY sample_data ./sample_data

WORKDIR /app
COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen


EXPOSE 8000 8001 8002

