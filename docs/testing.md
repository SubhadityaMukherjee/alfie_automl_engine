# Testing

## Run all tests
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