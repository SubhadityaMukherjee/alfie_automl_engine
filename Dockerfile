# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV PATH="/app/.venv/bin:$PATH"

# --------------------
# Copy lockfiles first (cache-friendly)
# --------------------
COPY pyproject.toml uv.lock ./

# --------------------
# Install dependencies only
# --------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# --------------------
# Copy application code
# --------------------
COPY app/ app/

# --------------------
# Install project itself
# --------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
RUN chown -R nonroot:nonroot /app
USER nonroot
ENTRYPOINT []

