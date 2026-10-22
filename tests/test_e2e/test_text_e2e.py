"""End-to-end tests for the text AutoML pipeline.

One POST runs the full text stack: router → orchestrator → shared ZIP
extraction → text validation → consolidated ML engine (real Optuna search +
Lightning Fabric training on a tiny local BERT) → artifact packaging → AutoDW
upload payload. Only the AutoDW HTTP boundary and the Hugging Face hub are
stubbed (see ``tests/test_e2e/conftest.py``).

Happy-path tests train real models and are marked ``full``; the failure-path
tests run the real pipeline up to the rejection point and are cheap enough
for the default suite.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_e2e.artifacts import text_vocab, zip_entry_names, zip_member_json

client = TestClient(app)

_TEXT_PARAMS = {
    "user_id": "user1",
    "dataset_id": "ds-text",
    "text_column": "text",
    "label_column": "label",
    "task_type": "text_classification",
    "time_budget": 60,  # -> exactly 1 Optuna trial
    "model_size": "small",
}

_EXPECTED_STEPS = [
    "request_received",
    "fetch_metadata",
    "validate_dataset",
    "resolve_download_url",
    "download_and_extract",
    "validate_inputs",
    "train",
    "serialize_model",
    "build_upload_payload",
    "upload_model",
]


@pytest.mark.full
def test_text_classification_end_to_end(stub_autodw, text_zip, zip_metadata):
    """text_classification: train on a real CSV and package artifacts."""
    calls = stub_autodw("text", text_zip, zip_metadata)

    resp = client.post("/automl/text/best_model/", data=_TEXT_PARAMS)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert "Text AutoML training completed" in body["message"]

    leaderboard = json.loads(body["leaderboard"])
    assert leaderboard["trials"] == 1
    assert isinstance(leaderboard["best_loss"], float)
    assert "loss" in leaderboard["best_metrics"]

    steps = [e["step"] for e in body["process_log"] if e["type"] == "step"]
    assert steps == _EXPECTED_STEPS
    assert all(e["status"] == "ok" for e in body["process_log"])

    # The archive carries a trained model, the label map, and the tokenizer
    # vocabulary needed to rebuild the preprocessing pipeline.
    assert calls["zip_bytes"], "upload stub received an empty archive"
    assert "feature_mapping.json" in zip_entry_names(calls["zip_bytes"])
    assert "model.pt" in zip_entry_names(calls["zip_bytes"])

    mapping = zip_member_json(calls["zip_bytes"], "feature_mapping.json")
    assert mapping["task_type"] == "text_classification"
    assert mapping["label_map"]["id2label"] == {"0": "neg", "1": "pos"}
    assert mapping["label_map"]["label2id"] == {"neg": 0, "pos": 1}

    tokenizer_mapping = mapping["tokenizer"]
    assert tokenizer_mapping["vocab"] == text_vocab()

    payload = calls["payload"]
    assert payload["model_type"] == "text_classification"
    assert payload["training_dataset"] == "ds-text"
    assert payload["framework"] == "pytorch"
    assert payload["model_id"].startswith("text_automl_ds-text_")
    assert calls["upload_url"].endswith("/ai-models/upload/single/user1")


def test_text_classification_unsupported_task_rejected_end_to_end(
    stub_autodw, text_zip, zip_metadata
):
    """A non-text task_type is rejected by the orchestrator's task check.

    The real pipeline still downloads and extracts the ZIP first.
    """
    stub_autodw("text", text_zip, zip_metadata)

    params = {**_TEXT_PARAMS, "task_type": "image_classification"}
    resp = client.post("/automl/text/best_model/", data=params)
    assert resp.status_code == 400
    assert "Unsupported task_type" in resp.json()["error"]

    body = resp.json()
    statuses = {
        e["step"]: e["status"] for e in body["process_log"] if e["type"] == "step"
    }
    assert statuses["download_and_extract"] == "ok"
    assert "train" not in statuses


def test_text_classification_missing_column_rejected_end_to_end(
    stub_autodw, zip_metadata
):
    """A CSV without the requested text column is rejected by validation."""
    from tests.test_e2e.artifacts import make_zip_bytes

    entries = {
        "labels.csv": "sentence,label\ngrim dull days,neg\nbright calm days,pos\n",
        "images/.keep": "",
    }
    stub_autodw("text", make_zip_bytes(entries), zip_metadata)

    resp = client.post("/automl/text/best_model/", data=_TEXT_PARAMS)
    assert resp.status_code == 400
    assert "text" in resp.json()["error"].lower()

    statuses = {
        e["step"]: e["status"]
        for e in resp.json()["process_log"]
        if e["type"] == "step"
    }
    assert statuses["validate_inputs"] == "failed"
    assert "train" not in statuses
