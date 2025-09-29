# --------- Builder Stage ----------
FROM python:3.12-slim-trixie AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
    libgl1 libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv (dependency manager)
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/
WORKDIR /app

# Copy only dependency manifests for caching
COPY pyproject.toml uv.lock ./

# Create venv and install deps
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen

# --------- Runtime Stage ----------
FROM python:3.12-slim-trixie AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code (only once)
COPY app/ ./app
COPY sample_data ./sample_data

EXPOSE 8000 8001 8002

ENTRYPOINT ["uv", "run", "uvicorn"]
# Default (overridden in docker-compose)
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]

