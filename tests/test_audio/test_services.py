"""Tests for app/audio_automl/services.py."""

import json
import zipfile

import pandas as pd
import pytest

from app.audio_automl.services import (
    build_upload_payload,
    collect_non_audio_files,
    convert_leaderboard_safely,
    serialize_and_zip_model,
    validate_audio_inputs,
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
# validate_audio_inputs
# ---------------------------------------------------------------------------


def _make_audio_dataset(tmp_path, audio_dir_name="audio"):
    audio_dir = tmp_path / audio_dir_name
    audio_dir.mkdir()
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"audio_path": ["a.wav"], "label": ["cat"]}).to_csv(
        csv_path, index=False
    )
    return csv_path, audio_dir


def test_validate_audio_inputs_valid(tmp_path):
    csv_path, audio_dir = _make_audio_dataset(tmp_path)
    result = validate_audio_inputs(csv_path, audio_dir, "audio_path", "label")
    assert result is None


def test_validate_audio_inputs_missing_audio_dir(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"audio_path": ["a.wav"], "label": ["cat"]}).to_csv(
        csv_path, index=False
    )
    result = validate_audio_inputs(
        csv_path, tmp_path / "nonexistent_audio", "audio_path", "label"
    )
    assert result is not None
    assert "Audio directory" in result


def test_validate_audio_inputs_missing_csv(tmp_path):
    _, audio_dir = _make_audio_dataset(tmp_path)
    result = validate_audio_inputs(
        tmp_path / "nonexistent.csv", audio_dir, "audio_path", "label"
    )
    assert result is not None
    assert "Labels CSV not found" in result


def test_validate_audio_inputs_missing_columns(tmp_path):
    csv_path, audio_dir = _make_audio_dataset(tmp_path)
    result = validate_audio_inputs(csv_path, audio_dir, "wrong_col", "label")
    assert result is not None
    assert "wrong_col" in result
    assert "Filename" in result


def test_validate_audio_inputs_rejects_non_audio_files(tmp_path):
    """Files in the CSV that are not audio clips are rejected before training."""
    csv_path, audio_dir = _make_audio_dataset(tmp_path)
    pd.DataFrame(
        {"audio_path": ["clip.wav", "img.png", "notes.txt"], "label": ["cat"] * 3}
    ).to_csv(csv_path, index=False)

    result = validate_audio_inputs(csv_path, audio_dir, "audio_path", "label")
    assert result is not None
    assert "non-audio" in result
    assert "img.png" in result
    assert "notes.txt" in result


def test_validate_audio_inputs_all_audio_extensions_pass(tmp_path):
    csv_path, audio_dir = _make_audio_dataset(tmp_path)
    names = ["a.wav", "b.MP3", "c.flac", "d.ogg", "e.m4a", "f.aac"]
    pd.DataFrame({"audio_path": names, "label": ["cat"] * len(names)}).to_csv(
        csv_path, index=False
    )

    assert validate_audio_inputs(csv_path, audio_dir, "audio_path", "label") is None


# ---------------------------------------------------------------------------
# collect_non_audio_files
# ---------------------------------------------------------------------------


def test_collect_non_audio_files_flags_wrong_extensions():
    df = pd.DataFrame(
        {"audio_path": ["a.wav", "b.png", "c.txt", "d.MP3"], "label": [0, 1, 0, 1]}
    )
    assert collect_non_audio_files(df, "audio_path") == ["b.png", "c.txt"]


def test_collect_non_audio_files_flags_missing_extension():
    df = pd.DataFrame({"audio_path": ["README"], "label": [0]})
    assert collect_non_audio_files(df, "audio_path") == ["README"]


def test_collect_non_audio_files_all_valid():
    df = pd.DataFrame({"audio_path": ["a.wav", "b.flac", "c.ogg"], "label": [0, 1, 0]})
    assert collect_non_audio_files(df, "audio_path") == []


def test_validate_audio_inputs_rejects_non_audio_task(tmp_path):
    csv_path, audio_dir = _make_audio_dataset(tmp_path)
    result = validate_audio_inputs(
        csv_path, audio_dir, "audio_path", "label", task_type="image_classification"
    )
    assert result is not None
    assert "image_classification" in result


# ---------------------------------------------------------------------------
# convert_leaderboard_safely
# ---------------------------------------------------------------------------


def test_convert_leaderboard_safely_complete():
    result = {
        "best_value": 0.123,
        "best_params": {"lr": 0.001},
        "n_trials": 3,
    }
    lb_json, lb_str = convert_leaderboard_safely(result)
    assert lb_json["best_loss"] == 0.123
    assert lb_json["trials"] == 3
    assert json.loads(lb_str) == lb_json


# ---------------------------------------------------------------------------
# build_upload_payload
# ---------------------------------------------------------------------------


def test_build_upload_payload_model_id_prefix(fake_metadata):
    model_id, _ = build_upload_payload(
        "my_dataset", "v1", fake_metadata, "audio_classification", {}
    )
    assert model_id.startswith("audio_automl_my_dataset_")


def test_build_upload_payload_framework_is_pytorch(fake_metadata):
    _, data = build_upload_payload(
        "ds1", "v1", fake_metadata, "audio_classification", {}
    )
    assert data["framework"] == "pytorch"


# ---------------------------------------------------------------------------
# serialize_and_zip_model
# ---------------------------------------------------------------------------


def test_serialize_and_zip_model_creates_zip(tmp_path):
    zip_path = serialize_and_zip_model(tmp_path)
    assert zip_path.exists()
    assert zip_path.name == "audio_model.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "audio_deployment_instructions.md" in zf.namelist()
