
# Run the services

## Tabular AutoML API
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

## Vision AutoML and Website Accessibility
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
