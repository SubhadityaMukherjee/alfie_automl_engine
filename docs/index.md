# ALFIE AutoML Engine

Minimal FastAPI services and utilities for AutoML on tabular and vision data, plus a website accessibility toolkit.

## Quickstart

### Prerequisites
- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- bash and wget (for sample data script; on macOS: `brew install wget`)

### Environment and dependencies (recommended: uv)
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

### Generate sample data (optional)
The repo includes `mk_sample_data.sh` which downloads small datasets under `sample_data/`:
```bash
# From project root
bash mk_sample_data.sh
```
What it fetches:
- `sample_data/knot_theory/{train.csv,test.csv}`
- `sample_data/m4_hourly_subset/{train.csv,test.csv}`

If `wget` is missing on macOS: `brew install wget`.

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

## Configuration
- Copy the `.env.template` to `.env` and fill in whatever is missing
- `DATABASE_URL` (optional): SQLAlchemy connection string. Default is `sqlite:///automl_sessions.db` created at repo root.
- Change the ports if needed
- Uploads are saved under `uploaded_data/`.
- AutoML artifacts (from training) are written alongside the uploaded session folder in `automl_data_path/`.

You can set environment variables via the `.env` file in the project root.
