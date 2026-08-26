"""Tests for app/text_automl/services.py."""

import json
import zipfile

import pandas as pd
import pytest

from app.text_automl.services import (
    build_upload_payload,
    convert_leaderboard_safely,
    serialize_and_zip_model,
    validate_text_inputs,
)


@pytest.fixture
def fake_metadata():
    return {
        "file_type": "zip",
        "original_filename": "dataset.zip",
        "version": "v1",
        "custom_metadata": {},
    }


# ---------------------------------------------------------------------------
# validate_text_inputs
# ---------------------------------------------------------------------------


def test_validate_text_inputs_classification_valid(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"text": ["hello world"], "label": ["pos"]}).to_csv(
        csv_path, index=False
    )
    assert validate_text_inputs(csv_path, "text_classification") is None


def test_validate_text_inputs_classification_custom_text_column(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"sentence": ["hello"], "label": ["pos"]}).to_csv(
        csv_path, index=False
    )
    assert (
        validate_text_inputs(csv_path, "text_classification", text_column="sentence")
        is None
    )
    result = validate_text_inputs(csv_path, "text_classification", text_column="text")
    assert result is not None
    assert "text_classification" in result


def test_validate_text_inputs_classification_missing_columns(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"sentence": ["hello"], "sentiment": ["pos"]}).to_csv(
        csv_path, index=False
    )
    result = validate_text_inputs(csv_path, "text_classification")
    assert result is not None
    assert "text_classification" in result


def test_validate_text_inputs_question_answering_valid(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "question": ["What?"],
            "context": ["Some text"],
            "answer_start": [0],
            "answer_text": ["Some"],
        }
    ).to_csv(csv_path, index=False)
    assert validate_text_inputs(csv_path, "question_answering") is None


def test_validate_text_inputs_question_answering_missing_columns(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"question": ["What?"], "context": ["Text"]}).to_csv(
        csv_path, index=False
    )
    result = validate_text_inputs(csv_path, "question_answering")
    assert result is not None
    assert "question_answering" in result


def test_validate_text_inputs_causal_lm_valid(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"text": ["once upon a time"]}).to_csv(csv_path, index=False)
    assert validate_text_inputs(csv_path, "causal_lm") is None


def test_validate_text_inputs_seq2seq_lm_missing_columns(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(csv_path, index=False)
    result = validate_text_inputs(csv_path, "seq2seq_lm")
    assert result is not None
    assert "seq2seq_lm" in result


def test_validate_text_inputs_unreadable_csv(tmp_path):
    result = validate_text_inputs(tmp_path / "nonexistent.csv", "causal_lm")
    assert result is not None
    assert "Could not read labels CSV" in result


def test_validate_text_inputs_rejects_non_text_task(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(csv_path, index=False)
    result = validate_text_inputs(csv_path, "image_classification")
    assert result is not None
    assert "image_classification" in result


# ---------------------------------------------------------------------------
# convert_leaderboard_safely
# ---------------------------------------------------------------------------


def test_convert_leaderboard_safely_complete():
    result = {
        "best_value": 0.05,
        "best_params": {"lr": 0.01},
        "n_trials": 2,
    }
    lb_json, lb_str = convert_leaderboard_safely(result)
    assert lb_json["best_loss"] == 0.05
    assert json.loads(lb_str) == lb_json


# ---------------------------------------------------------------------------
# build_upload_payload
# ---------------------------------------------------------------------------


def test_build_upload_payload_model_id_prefix(fake_metadata):
    model_id, _ = build_upload_payload(
        "my_dataset", "v1", fake_metadata, "text_classification", {}
    )
    assert model_id.startswith("text_automl_my_dataset_")


def test_build_upload_payload_framework_is_pytorch(fake_metadata):
    _, data = build_upload_payload(
        "ds1", "v1", fake_metadata, "text_classification", {}
    )
    assert data["framework"] == "pytorch"


# ---------------------------------------------------------------------------
# serialize_and_zip_model
# ---------------------------------------------------------------------------


def test_serialize_and_zip_model_creates_zip(tmp_path):
    zip_path = serialize_and_zip_model(tmp_path)
    assert zip_path.exists()
    assert zip_path.name == "text_model.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "text_deployment_instructions.md" in zf.namelist()
