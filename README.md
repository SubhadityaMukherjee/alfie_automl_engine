# ALFIE AutoML Engine

Minimal FastAPI services and utilities for AutoML on tabular and vision data, plus a website accessibility toolkit.

## Quickstart

### Prerequisites
- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Create a virtual environment
```bash
pip install uv
uv venv --seed --python 3.11 ~/.venvs/alfieautoml
source ~/.venvs/alfieautoml/bin/activate
```

### Install dependencies
With uv (fast, uses lockfile if present):
```bash
uv pip sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

## Configuration
- `DATABASE_URL` (optional): SQLAlchemy connection string. Default is `sqlite:///automl_sessions.db` created at repo root.
- Uploads are saved under `uploaded_data/`.
- AutoML artifacts (from training) are written alongside the uploaded session folder in `automl_data_path/`.

You can set environment variables inline when running commands, or via a `.env` file in the project root.

## Run the services

### Tabular AutoML API
Start the FastAPI app:
```bash
uvicorn app.tabular_automl.main:app --reload
```

Endpoints:
- `POST /automl_tabular/get_user_input/`
  - Form fields: `target_column_name`, `task_type` (classification|regression|time series), `time_budget` (int), optional `time_stamp_column_name`.
  - Files: `train_file` (required), `test_file` (optional).
  - On success: stores the session in DB and returns `{ session_id }`.

- `POST /automl_tabular/find_best_model/`
  - JSON body: `{ "session_id": "..." }`.
  - Loads stored session, trains models, returns leaderboard (markdown string or raw content).

Example using curl with sample data:
```bash
curl -X POST http://127.0.0.1:8000/automl_tabular/get_user_input/ \
  -F "train_file=@sample_data/frames.csv" \
  -F "target_column_name=target" \
  -F "task_type=classification" \
  -F "time_budget=60"

curl -X POST http://127.0.0.1:8000/automl_tabular/find_best_model/ \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<returned-session-id>"}'
```

### Vision AutoML and Website Accessibility
There are additional modules under `app/vision_automl/` and `app/website_accessibility/`. Each contains a `main.py`, `services.py`, and supporting modules. Start them similarly if they expose FastAPI apps, e.g.:
```bash
uvicorn app.website_accessibility.main:app --reload --port 8002
```

Note: Some modules may rely on larger ML frameworks; ensure GPU drivers and frameworks are installed if required.

## Database
- Model: `app/tabular_automl/db.py` defines `AutoMLSession` and initializes the DB on import.
- Default SQLite DB file is created as `automl_sessions.db` at the repo root.
- To use a different database, set `DATABASE_URL`, for example:
```bash
export DATABASE_URL="sqlite:///./local.db"
```

## Testing

### Run all tests
```bash
pytest -q
```

### Run fast subset (default) vs full suite
The test suite skips long-running tests by default. To run the full suite:
```bash
pytest -q --full
```

### Run specific tests/files
```bash
pytest -q tests/tabular_automl/test_modules.py::test_trainer_init
pytest -q tests/tabular_automl/test_services.py
```

### Notes
- Tests isolate filesystem operations using `tmp_path` and mock/monkeypatch database sessions to an in-memory SQLite engine.
- HTTP endpoint tests use `fastapi.testclient.TestClient` and monkeypatch service-layer functions to avoid heavy training runs.

## Development
- Useful scripts:
  - `mk_sample_data.sh`: helpers for sample data.
  - `test_services.sh`: quick test runner.
<!-- 
## Docker (optional)
There are `Dockerfile` and `docker_compose.yaml` files in the `app/` subfolders for containerized runs. Build and run as needed, e.g.:
```bash
docker build -t alfie-tabular app/tabular_automl
docker run --rm -p 8000:8000 alfie-tabular
``` -->
