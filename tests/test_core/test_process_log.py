"""Tests for the per-request process log."""

import logging
from contextvars import copy_context

import pytest

from app.core.process_log import (
    ProcessLogHandler,
    get_process_log,
    log_step,
    start_process_log,
    step,
)


@pytest.fixture(autouse=True)
def _reset_process_log():
    """Ensure each test starts without an active process log."""
    start_process_log()
    get_process_log().clear()
    yield


@pytest.fixture()
def capture_handler():
    """Attach exactly one ProcessLogHandler to the root logger for the test."""
    handler = ProcessLogHandler()
    root = logging.getLogger()
    old_level = root.level
    existing = [h for h in root.handlers if isinstance(h, ProcessLogHandler)]
    for h in existing:
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield handler
    root.removeHandler(handler)
    for h in existing:
        root.addHandler(h)
    root.setLevel(old_level)


def test_get_process_log_empty_when_not_started():
    def run():
        # Simulate a context where no log was started
        from app.core import process_log

        process_log._entries.set(None)  # noqa: SLF001 - test-only reset
        assert get_process_log() == []
        log_step("anything")  # must be a no-op

    copy_context().run(run)


def test_start_process_log_records_request_start():
    start_process_log("task-123")
    entries = get_process_log()
    assert len(entries) == 1
    assert entries[0]["step"] == "request_received"
    assert entries[0]["status"] == "ok"
    assert entries[0]["task_id"] == "task-123"
    assert entries[0]["type"] == "step"


def test_start_process_log_without_task_id():
    start_process_log()
    assert "task_id" not in get_process_log()[0]


def test_log_step_details_and_none_dropping():
    start_process_log()
    log_step("train", "ok", n_trials=3, error=None)
    entry = get_process_log()[-1]
    assert entry["step"] == "train"
    assert entry["status"] == "ok"
    assert entry["n_trials"] == 3
    assert "error" not in entry
    assert "timestamp" in entry


def test_step_context_manager_success():
    start_process_log()
    with step("download"):
        pass
    entry = get_process_log()[-1]
    assert entry["type"] == "step"
    assert entry["step"] == "download"
    assert entry["status"] == "ok"


def test_step_context_manager_failure():
    start_process_log()
    with pytest.raises(ValueError, match="boom"):
        with step("train"):
            raise ValueError("boom")
    entry = get_process_log()[-1]
    assert entry["step"] == "train"
    assert entry["status"] == "failed"
    assert "boom" in entry["error"]


def test_step_noop_without_started_log():
    # Must not raise even when no process log is active
    with step("train"):
        pass


def test_contexts_are_isolated():
    results = {}

    def run(name):
        start_process_log(name)
        log_step("s1")
        results[name] = len(get_process_log())

    ctx_a = copy_context()
    ctx_b = copy_context()
    ctx_a.run(run, "a")
    ctx_b.run(run, "b")
    # Each context saw only its own request_received + s1 entries
    assert results["a"] == 2
    assert results["b"] == 2


def test_handler_captures_app_info_records(capture_handler):
    start_process_log()
    app_logger = logging.getLogger("app.some_service")
    app_logger.info("training finished")
    app_logger.error("upload failed: %s", "timeout")

    log_entries = [
        e
        for e in get_process_log()
        if e["type"] == "log" and e["logger"] == "app.some_service"
    ]
    assert [e["message"] for e in log_entries] == [
        "training finished",
        "upload failed: timeout",
    ]
    assert log_entries[0]["level"] == "INFO"
    assert log_entries[1]["level"] == "ERROR"


def test_handler_ignores_non_app_and_debug_records(capture_handler):
    start_process_log()
    logging.getLogger("autogluon.trainer").warning("noise")
    logging.getLogger("optuna").error("noise")
    logging.getLogger("app.some_service").debug("too verbose")

    assert [e for e in get_process_log() if e["type"] == "log"] == []


def test_handler_noop_without_started_log(capture_handler):
    def run():
        from app.core import process_log

        process_log._entries.set(None)  # noqa: SLF001 - test-only reset
        logging.getLogger("app.some_service").error("should be ignored")

    copy_context().run(run)


def test_handler_truncates_after_cap(capture_handler):
    from app.core.process_log import _MAX_LOG_ENTRIES

    start_process_log()
    noisy = logging.getLogger("app.noisy")
    for i in range(_MAX_LOG_ENTRIES + 10):
        noisy.info("line %d", i)

    log_entries = [e for e in get_process_log() if e["type"] == "log"]
    assert len(log_entries) == _MAX_LOG_ENTRIES + 1  # cap + truncation notice
    assert "truncated" in log_entries[-1]["message"]

    # Step markers are still appended after truncation
    log_step("still_here")
    assert get_process_log()[-1]["step"] == "still_here"


# ---------------------------------------------------------------------------
# Integration: process_log in the actual HTTP payloads
# ---------------------------------------------------------------------------


def test_error_payload_contains_failed_step():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post(
        "/automl/tabular/best_model/",
        data={
            "user_id": "",
            "dataset_id": "ds1",
            "target_column_name": "target",
            "task_type": "classification",
            "time_budget": "10",
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    steps = [e for e in body["process_log"] if e["type"] == "step"]
    assert steps[0]["step"] == "request_received"
    failed = steps[-1]
    assert failed["step"] == "validate_request"
    assert failed["status"] == "failed"
    assert "user_id" in failed["error"]


def test_automlplus_error_payload_contains_process_log():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post(
        "/automl/automl_plus/image_tools/run_on_image/", data={"prompt": "describe"}
    )
    assert resp.status_code == 400
    assert resp.json()["process_log"][0]["step"] == "request_received"
