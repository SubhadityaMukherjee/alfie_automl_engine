# Testing

## Run all tests
```bash
uv run pytest -q
```

### Run fast subset (default) vs full suite
The test suite skips long-running tests by default. To run the full suite:
```bash
uv run pytest -q --full
```

### Run specific tests/files
The test tree mirrors the app layout (`tests/test_ml_engine`, `tests/test_tabular`,
`tests/test_vision`, `tests/test_audio`, `tests/test_text`, `tests/test_core`,
`tests/test_automlplus`, `tests/test_integration`):
```bash
uv run pytest -q tests/test_tabular/test_automl_trainer.py::test_trainer_init
uv run pytest -q tests/test_ml_engine
```

### Notes
- Tests isolate filesystem operations using `tmp_path`.
- HTTP endpoint tests use `fastapi.testclient.TestClient` and patch
  orchestrator-level functions to avoid heavy training runs.

## Development
- Useful scripts:
  - `mk_sample_data.sh`: helpers for sample data.
  - `test_services.sh`: quick test runner.
  - `test_services.py`: curl-based smoke checks against a locally running engine.
- Lint / types (also run via pre-commit):
```bash
uv run ruff check app tests
uv run ruff format --check app tests
uv run mypy app/
```
