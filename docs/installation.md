# Installation Instructions

## Prerequisites
- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- bash and wget (for sample data script; on macOS: `brew install wget`)

## Environment and dependencies (recommended: uv)
```bash
# Install uv (user scope)
pip install --user uv

# From project root, create and activate a virtual env
uv venv --seed --python 3.11 .venv
source .venv/bin/activate

# Install project dependencies
# Option A: Use pyproject + uv.lock (preferred if `uv.lock` is present)
uv sync --frozen

# Option B: Sync from requirements.txt
uv pip sync requirements.txt
```

Alternative with pip only:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## !!AutoDW
- Another important package we need is AutoDW, follow [this](autodw.md) for instructions

## Managing packages with uv

Use uv to manage dependencies declared in `pyproject.toml` (lockfile: `uv.lock`). Common tasks:
```bash
# Add a runtime dependency (updates pyproject and lockfile)
uv add requests

# Remove a dependency
uv remove requests

# Upgrade all dependencies to latest allowed by constraints
uv lock --upgrade
uv sync

# Install dev-only tools (use --dev group if you have groups defined)
uv add --dev ruff
```


