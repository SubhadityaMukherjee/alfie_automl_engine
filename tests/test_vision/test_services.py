"""Tests for app/vision_automl/services.py."""

import json
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from app.core.exceptions import AutoDWDownloadError
from app.core.service_helpers import build_metadata_url as _build_metadata_url
from app.vision_automl.services import (
    build_upload_payload,
    collect_non_image_files,
    convert_leaderboard_safely,
    download_dataset,
    fetch_dataset_metadata,
    resolve_download_url,
    serialize_and_zip_model,
    validate_vision_inputs,
)

# ---------------------------------------------------------------------------
# _build_metadata_url
# ---------------------------------------------------------------------------


def test_build_metadata_url_without_version():
    url = _build_metadata_url("http://autodw", "user1", "ds1", None)
    assert url == "http://autodw/datasets/user1/ds1"


def test_build_metadata_url_with_version():
    url = _build_metadata_url("http://autodw", "user1", "ds1", "v2")
    assert url == "http://autodw/datasets/user1/ds1/version/v2"


def test_build_metadata_url_empty_version_treated_as_no_version():
    url = _build_metadata_url("http://autodw", "u", "d", "")
    assert url == "http://autodw/datasets/u/d"


# ---------------------------------------------------------------------------
# fetch_dataset_metadata
# ---------------------------------------------------------------------------


@patch("app.core.service_helpers.requests.get")
def test_fetch_dataset_metadata_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"file_type": "zip"}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp
    result = fetch_dataset_metadata("http://base", "u1", "d1", None)
    assert result == {"file_type": "zip"}
    mock_get.assert_called_once()


@patch("app.core.service_helpers.requests.get")
def test_fetch_dataset_metadata_raises_on_http_error(mock_get):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
    mock_get.return_value = mock_resp
    with pytest.raises(AutoDWDownloadError):
        fetch_dataset_metadata("http://base", "u1", "d1", None)


# ---------------------------------------------------------------------------
# resolve_download_url
# ---------------------------------------------------------------------------


def test_resolve_download_url_no_split(fake_metadata):
    url = resolve_download_url("http://base", "u", "d", None, fake_metadata, None)
    assert url == "http://base/datasets/u/d/download"


def test_resolve_download_url_with_split_and_metadata_has_split(
    fake_metadata_with_splits,
):
    url = resolve_download_url(
        "http://base", "u", "d", None, fake_metadata_with_splits, "train"
    )
    assert url == "http://base/datasets/u/d/download?split=train"


def test_resolve_download_url_split_requested_but_no_metadata_split(
    fake_metadata, caplog
):
    import logging

    with caplog.at_level(logging.WARNING):
        url = resolve_download_url(
            "http://base", "u", "d", None, fake_metadata, "train"
        )
    assert "?split=" not in url
    assert "no splits" in caplog.text or "downloading full" in caplog.text.lower()


def test_resolve_download_url_with_version():
    metadata = {"custom_metadata": {}}
    url = resolve_download_url("http://base", "u", "d", "v2", metadata, None)
    assert "/version/v2/download" in url


# ---------------------------------------------------------------------------
# download_dataset
# ---------------------------------------------------------------------------


@patch("app.core.service_helpers.requests.get")
def test_download_dataset_writes_zip_file(mock_get, tmp_path):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_get.return_value.__enter__ = lambda s: mock_resp
    mock_get.return_value.__exit__ = MagicMock(return_value=False)

    result = download_dataset("http://example.com/dl", tmp_path, "dataset.zip")
    assert result == tmp_path / "dataset.zip"
    assert (tmp_path / "dataset.zip").exists()
    assert (tmp_path / "dataset.zip").read_bytes() == b"chunk1chunk2"


@patch("app.core.service_helpers.requests.get")
def test_download_dataset_raises_on_http_error(mock_get, tmp_path):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("503")
    mock_get.return_value.__enter__ = lambda s: mock_resp
    mock_get.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(AutoDWDownloadError):
        download_dataset("http://example.com/dl", tmp_path, "dataset.zip")


# ---------------------------------------------------------------------------
# validate_vision_inputs
# ---------------------------------------------------------------------------


def test_validate_vision_inputs_valid(synthetic_images_dir):
    csv_path, images_dir = synthetic_images_dir
    result = validate_vision_inputs(csv_path, images_dir, "filename", "label")
    assert result is None


def test_validate_vision_inputs_missing_filename_column(synthetic_images_dir):
    csv_path, images_dir = synthetic_images_dir
    result = validate_vision_inputs(csv_path, images_dir, "wrong_col", "label")
    assert result is not None
    assert "wrong_col" in result
    assert "Filename" in result


def test_validate_vision_inputs_missing_label_column(synthetic_images_dir):
    csv_path, images_dir = synthetic_images_dir
    result = validate_vision_inputs(csv_path, images_dir, "filename", "wrong_label")
    assert result is not None
    assert "wrong_label" in result
    assert "Label" in result


def test_validate_vision_inputs_unreadable_csv(tmp_path):
    result = validate_vision_inputs(
        tmp_path / "nonexistent.csv", tmp_path, "filename", "label"
    )
    assert result is not None
    assert "Could not read labels CSV" in result


def test_validate_vision_inputs_missing_images(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (10, 10)).save(images_dir / "img0.png")

    df = pd.DataFrame(
        {
            "filename": ["img0.png", "missing1.png", "missing2.png"],
            "label": ["cat", "cat", "dog"],
        }
    )
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)

    result = validate_vision_inputs(csv_path, images_dir, "filename", "label")
    assert result is not None
    assert "Missing" in result
    assert "2" in result


def test_validate_vision_inputs_many_missing_files_truncated(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    df = pd.DataFrame(
        {
            "filename": [f"missing{i}.png" for i in range(10)],
            "label": ["cat"] * 10,
        }
    )
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    result = validate_vision_inputs(csv_path, images_dir, "filename", "label")
    assert result is not None
    assert "..." in result


def test_validate_vision_inputs_rejects_audio_task_type(synthetic_images_dir):
    """Audio/text task types moved to their own endpoints; vision rejects them."""
    csv_path, images_dir = synthetic_images_dir
    result = validate_vision_inputs(
        csv_path, images_dir, "filename", "label", task_type="audio_classification"
    )
    assert result is not None
    assert "audio_classification" in result


def test_validate_vision_inputs_rejects_text_task_type(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"text": ["hello"], "label": ["pos"]}).to_csv(csv_path, index=False)
    result = validate_vision_inputs(
        csv_path, tmp_path, "text", "label", task_type="text_classification"
    )
    assert result is not None
    assert "text_classification" in result


def test_validate_vision_inputs_non_image_files(tmp_path):
    """Files in the CSV that are not images are rejected before training."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "filename": ["clip.wav", "notes.txt", "img.png"],
            "label": ["cat", "cat", "dog"],
        }
    ).to_csv(csv_path, index=False)

    result = validate_vision_inputs(csv_path, images_dir, "filename", "label")
    assert result is not None
    assert "non-image" in result
    assert "clip.wav" in result
    assert "notes.txt" in result


def test_validate_vision_inputs_all_image_extensions_pass(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    from PIL import Image

    names = ["a.png", "b.JPG", "c.JPEG", "d.bmp", "e.gif", "f.webp", "g.tiff"]
    for name in names:
        Image.new("RGB", (4, 4)).save(images_dir / name)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"filename": names, "label": ["cat"] * len(names)}).to_csv(
        csv_path, index=False
    )

    assert validate_vision_inputs(csv_path, images_dir, "filename", "label") is None


# ---------------------------------------------------------------------------
# collect_non_image_files
# ---------------------------------------------------------------------------


def test_collect_non_image_files_flags_wrong_extensions():
    df = pd.DataFrame(
        {"filename": ["a.png", "b.wav", "c.txt", "d.JPG"], "label": [0, 1, 0, 1]}
    )
    result = collect_non_image_files(df, "filename")
    assert result == ["b.wav", "c.txt"]


def test_collect_non_image_files_flags_missing_extension():
    df = pd.DataFrame({"filename": ["README"], "label": [0]})
    assert collect_non_image_files(df, "filename") == ["README"]


def test_collect_non_image_files_all_valid():
    df = pd.DataFrame({"filename": ["a.png", "b.jpeg", "c.webp"], "label": [0, 1, 0]})
    assert collect_non_image_files(df, "filename") == []


def test_validate_vision_inputs_object_detection_missing_annotation_columns(
    synthetic_images_dir,
):
    csv_path, images_dir = synthetic_images_dir
    # synthetic_images_dir CSV has filename + label but NOT boxes/class_labels
    result = validate_vision_inputs(
        csv_path,
        images_dir,
        "filename",
        "label",
        task_type="object_detection",
    )
    assert result is not None
    assert "object_detection" in result


def test_validate_vision_inputs_object_detection_with_annotation_columns(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    from PIL import Image

    img_path = images_dir / "img0.png"
    Image.new("RGB", (10, 10)).save(img_path)

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "filename": ["img0.png"],
            "label": ["object"],
            "boxes": ["[[0,0,5,5]]"],
            "class_labels": ["[0]"],
        }
    ).to_csv(csv_path, index=False)

    result = validate_vision_inputs(
        csv_path,
        images_dir,
        "filename",
        "label",
        task_type="object_detection",
    )
    assert result is None


# ---------------------------------------------------------------------------
# convert_leaderboard_safely
# ---------------------------------------------------------------------------


def test_convert_leaderboard_safely_complete(fake_optuna_result):
    lb_json, lb_str = convert_leaderboard_safely(fake_optuna_result)
    assert lb_json["best_loss"] == 0.123
    assert lb_json["best_params"] == {"lr": 0.001, "model_id": "google/efficientnet-b0"}
    assert lb_json["trials"] == 3
    parsed = json.loads(lb_str)
    assert parsed == lb_json


def test_convert_leaderboard_safely_empty_dict():
    lb_json, lb_str = convert_leaderboard_safely({})
    assert lb_json["best_loss"] is None
    assert lb_json["best_params"] is None
    assert lb_json["trials"] is None
    assert json.loads(lb_str) == lb_json


# ---------------------------------------------------------------------------
# build_upload_payload
# ---------------------------------------------------------------------------


def test_build_upload_payload_returns_tuple(fake_metadata):
    model_id, data = build_upload_payload(
        "ds1", "v1", fake_metadata, "classification", {"best_loss": 0.1}
    )
    assert isinstance(model_id, str)
    assert isinstance(data, dict)


def test_build_upload_payload_model_id_prefix(fake_metadata):
    model_id, _ = build_upload_payload(
        "my_dataset", "v1", fake_metadata, "classification", {}
    )
    assert model_id.startswith("vision_automl_my_dataset_")


def test_build_upload_payload_framework_is_pytorch(fake_metadata):
    _, data = build_upload_payload("ds1", "v1", fake_metadata, "classification", {})
    assert data["framework"] == "pytorch"


def test_build_upload_payload_task_type_stored(fake_metadata):
    _, data = build_upload_payload("ds1", "v1", fake_metadata, "segmentation", {})
    assert data["model_type"] == "segmentation"


def test_build_upload_payload_explicit_version(fake_metadata):
    _, data = build_upload_payload("ds1", "v2", fake_metadata, "classification", {})
    assert data["training_dataset_version"] == "v2"


def test_build_upload_payload_none_version_uses_metadata(fake_metadata):
    _, data = build_upload_payload("ds1", None, fake_metadata, "classification", {})
    assert data["training_dataset_version"] == "v1"


def test_build_upload_payload_leaderboard_is_valid_json(fake_metadata):
    lb = {"best_loss": 0.05}
    _, data = build_upload_payload("ds1", "v1", fake_metadata, "classification", lb)
    parsed = json.loads(data["leaderboard"])
    assert parsed == lb


# ---------------------------------------------------------------------------
# serialize_and_zip_model
# ---------------------------------------------------------------------------


def test_serialize_and_zip_model_creates_zip(tmp_path):
    zip_path = serialize_and_zip_model(tmp_path)
    assert zip_path.exists()
    assert zip_path.suffix == ".zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.testzip()  # returns None if valid


def test_serialize_and_zip_model_includes_model_dir_contents(tmp_path):
    """Files placed in workdir/model/ by run_optuna_search should land in the zip."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "feature_mapping.json").write_text('{"task_type": "x"}')
    (model_dir / "model.pt").write_text("not really a torch file")

    zip_path = serialize_and_zip_model(tmp_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    assert "feature_mapping.json" in names
    assert "model.pt" in names
    assert "vision_deployment_instructions.md" in names
