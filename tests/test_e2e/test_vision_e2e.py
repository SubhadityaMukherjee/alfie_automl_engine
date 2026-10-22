"""End-to-end tests for the vision AutoML pipeline (image + multimodal).

A single POST travels the full stack: router → orchestrator → shared ZIP
extraction → vision validation → consolidated ML engine (real Optuna search
+ Lightning Fabric training on a tiny local ViT) → trial-artifact packaging →
AutoDW upload payload. Only the AutoDW HTTP boundary and the Hugging Face hub
are stubbed (see ``tests/test_e2e/conftest.py``).

Happy-path tests train real models and are marked ``full``; the failure-path
test runs the real pipeline up to the rejection point and is cheap enough for
the default suite.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_e2e.artifacts import zip_entry_names, zip_member_json

client = TestClient(app)

_VISION_PARAMS = {
    "user_id": "user1",
    "dataset_id": "ds-vision",
    "filename_column": "filename",
    "label_column": "label",
    "task_type": "image_classification",
    "time_budget": 60,  # -> exactly 1 Optuna trial
    "model_size": "small",
}

_MULTIMODAL_PARAMS = {
    "user_id": "user1",
    "dataset_id": "ds-multimodal",
    "filename_column": "filename",
    "label_column": "label",
    "time_budget": 60,
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
def test_image_classification_end_to_end(stub_autodw, vision_zip, zip_metadata):
    """image_classification: train, package artifacts, and build the upload."""
    calls = stub_autodw("vision", vision_zip, zip_metadata)

    resp = client.post("/automl/vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Success envelope: AutoDW-assigned identity used for retraining.
    assert "Vision AutoML training completed" in body["message"]
    assert "deployment_instructions" in body
    assert body["model_id"].startswith("vision_automl_ds-vision_")
    assert body["version"] == "v1"

    # Leaderboard: one completed trial with real metrics and params.
    leaderboard = json.loads(body["leaderboard"])
    assert leaderboard["trials"] == 1
    assert isinstance(leaderboard["best_loss"], float)
    assert "loss" in leaderboard["best_metrics"]
    assert leaderboard["best_params"]["model_id"]

    # Process log covers every pipeline stage, in order, without failures.
    steps = [e["step"] for e in body["process_log"] if e["type"] == "step"]
    assert steps == _EXPECTED_STEPS
    assert all(e["status"] == "ok" for e in body["process_log"])

    # The "uploaded" archive carries the training artifacts.
    assert calls["zip_bytes"], "upload stub received an empty archive"
    assert "feature_mapping.json" in zip_entry_names(calls["zip_bytes"])
    assert "model.pt" in zip_entry_names(calls["zip_bytes"])

    # Feature mapping captured the label vocabulary from the dataset.
    mapping = zip_member_json(calls["zip_bytes"], "feature_mapping.json")
    assert mapping["task_type"] == "image_classification"
    assert mapping["label_map"]["id2label"] == {"0": "cat", "1": "dog"}
    assert mapping["label_map"]["label2id"] == {"cat": 0, "dog": 1}

    # Upload payload describes the trained model for AutoDW.
    payload = calls["payload"]
    assert payload["model_type"] == "image_classification"
    assert payload["training_dataset"] == "ds-vision"
    assert payload["training_dataset_version"] == "v1"
    assert payload["framework"] == "pytorch"
    assert payload["model_id"].startswith("vision_automl_ds-vision_")
    # The reported model identity matches what was actually uploaded.
    assert payload["model_id"] == body["model_id"]
    assert calls["upload_url"].endswith("/ai-models/upload/single/user1")


@pytest.mark.full
def test_multimodal_classification_end_to_end(
    stub_autodw, multimodal_zip, zip_metadata
):
    """image_classification_multimodal: image + tabular fusion pipeline."""
    calls = stub_autodw("vision", multimodal_zip, zip_metadata)

    resp = client.post("/automl/vision/multimodal_best_model/", data=_MULTIMODAL_PARAMS)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert "Multimodal" in body["message"]
    # Aux columns auto-discovered from the CSV (all but filename + label).
    assert body["auxiliary_columns"] == ["weight", "color"]
    # AutoDW-assigned identity used for retraining.
    assert body["model_id"].startswith("vision_automl_ds-multimodal_")
    assert body["version"] == "v1"

    leaderboard = json.loads(body["leaderboard"])
    assert leaderboard["trials"] == 1

    steps = [e["step"] for e in body["process_log"] if e["type"] == "step"]
    assert steps == _EXPECTED_STEPS

    assert "feature_mapping.json" in zip_entry_names(calls["zip_bytes"])
    mapping = zip_member_json(calls["zip_bytes"], "feature_mapping.json")
    assert mapping["task_type"] == "image_classification_multimodal"

    aux = mapping["auxiliary_features"]
    assert aux["auxiliary_columns"] == ["weight", "color"]
    assert aux["numeric_columns"] == ["weight"]
    assert aux["categorical_columns"] == ["color"]
    assert aux["aux_feature_dim"] == 2
    assert "vision_embed_dim" in aux

    payload = calls["payload"]
    assert payload["model_type"] == "image_classification_multimodal"
    assert payload["training_dataset"] == "ds-multimodal"


def test_image_classification_missing_file_rejected_end_to_end(
    stub_autodw, broken_vision_zip, zip_metadata
):
    """CSV row referencing an absent image is rejected by real validation.

    Runs the true pipeline (real ZIP extraction + validation) up to the
    typed-validation failure — no validation code is mocked.
    """
    stub_autodw("vision", broken_vision_zip, zip_metadata)

    resp = client.post("/automl/vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert "Missing" in error and "image" in error

    body = resp.json()
    statuses = {
        e["step"]: e["status"] for e in body["process_log"] if e["type"] == "step"
    }
    assert statuses["download_and_extract"] == "ok"  # extraction really ran
    assert statuses["validate_inputs"] == "failed"  # validation rejected it
    assert "train" not in statuses  # training never started


def test_image_classification_non_zip_rejected_end_to_end(
    stub_autodw, vision_zip, zip_metadata
):
    """A CSV-only AutoDW dataset (no ZIP download) is rejected up front."""
    csv_metadata = {
        "file_type": "csv",
        "original_filename": "dataset.csv",
        "version": "v1",
        "custom_metadata": {},
    }
    stub_autodw("vision", vision_zip, csv_metadata)

    resp = client.post("/automl/vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 400
    assert "ZIP" in resp.json()["error"]
