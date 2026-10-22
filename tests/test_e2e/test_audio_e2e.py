"""End-to-end tests for the audio AutoML pipeline.

One POST runs the full audio stack: router → orchestrator → shared ZIP
extraction → audio validation → consolidated ML engine (real Optuna search +
Lightning Fabric training on a tiny local AST) → artifact packaging → AutoDW
upload payload. Only the AutoDW HTTP boundary and the Hugging Face hub are
stubbed (see ``tests/test_e2e/conftest.py``).

WAV decoding needs ``torchaudio`` (+ the ``soundfile`` backend), which the
engine ships as the optional ``audio`` extra — install it with
``uv sync --extra audio`` (or ``pip install .[audio]``). Without it the
training test skips cleanly; the validation test runs regardless because it
never reaches torchaudio.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_e2e.artifacts import zip_entry_names, zip_member_json

client = TestClient(app)

_AUDIO_PARAMS = {
    "user_id": "user1",
    "dataset_id": "ds-audio",
    "filename_column": "filename",
    "label_column": "label",
    "task_type": "audio_classification",
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
def test_audio_classification_end_to_end(stub_autodw, audio_zip, zip_metadata):
    """audio_classification: train on real WAV files and package artifacts."""
    pytest.importorskip(
        "torchaudio",
        reason="audio training needs the optional 'audio' extra: uv sync --extra audio",
    )
    calls = stub_autodw("audio", audio_zip, zip_metadata)

    resp = client.post("/automl/audio/best_model/", data=_AUDIO_PARAMS)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert "Audio AutoML training completed" in body["message"]
    # AutoDW-assigned identity used for retraining.
    assert body["model_id"].startswith("audio_automl_ds-audio_")
    assert body["version"] == "v1"

    leaderboard = json.loads(body["leaderboard"])
    assert leaderboard["trials"] == 1
    assert isinstance(leaderboard["best_loss"], float)
    assert "loss" in leaderboard["best_metrics"]

    steps = [e["step"] for e in body["process_log"] if e["type"] == "step"]
    assert steps == _EXPECTED_STEPS
    assert all(e["status"] == "ok" for e in body["process_log"])

    # The archive carries a trained model and the label vocabulary.
    assert calls["zip_bytes"], "upload stub received an empty archive"
    assert "feature_mapping.json" in zip_entry_names(calls["zip_bytes"])
    assert "model.pt" in zip_entry_names(calls["zip_bytes"])

    mapping = zip_member_json(calls["zip_bytes"], "feature_mapping.json")
    assert mapping["task_type"] == "audio_classification"
    assert mapping["label_map"]["id2label"] == {"0": "beep", "1": "hum"}
    assert mapping["label_map"]["label2id"] == {"beep": 0, "hum": 1}

    payload = calls["payload"]
    assert payload["model_type"] == "audio_classification"
    assert payload["training_dataset"] == "ds-audio"
    assert payload["framework"] == "pytorch"
    assert payload["model_id"].startswith("audio_automl_ds-audio_")
    # The reported model identity matches what was actually uploaded.
    assert payload["model_id"] == body["model_id"]
    assert calls["upload_url"].endswith("/ai-models/upload/single/user1")


def test_audio_classification_non_audio_file_rejected_end_to_end(
    stub_autodw, zip_metadata
):
    """Filenames without an audio extension are rejected by real validation.

    Runs the real pipeline (real ZIP extraction + validation); needs no
    torchaudio because training is never reached.
    """
    from tests.test_e2e.artifacts import make_zip_bytes

    entries = {
        "labels.csv": "filename,label\nclip_00.txt,beep\nclip_01.wav,hum\n",
        "images/clip_00.txt": "not audio",
        "images/clip_01.wav": b"RIFF....",  # never validated by content
    }
    stub_autodw("audio", make_zip_bytes(entries), zip_metadata)

    resp = client.post("/automl/audio/best_model/", data=_AUDIO_PARAMS)
    assert resp.status_code == 400
    assert "non-audio" in resp.json()["error"]

    body = resp.json()
    statuses = {
        e["step"]: e["status"] for e in body["process_log"] if e["type"] == "step"
    }
    assert statuses["download_and_extract"] == "ok"
    assert statuses["validate_inputs"] == "failed"
    assert "train" not in statuses
