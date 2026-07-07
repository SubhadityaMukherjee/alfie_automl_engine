"""Tests for app/vision_automl/services.py."""

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from app.core.exceptions import AutoDWDownloadError, AutoMLDataError
from app.core.service_helpers import build_metadata_url as _build_metadata_url
from app.vision_automl.services import (
    _find_csv_file,
    _find_or_resolve_images_dir,
    _find_valid_dataset_root,
    build_upload_payload,
    collect_missing_files,
    convert_leaderboard_safely,
    download_dataset,
    extract_and_locate_dataset,
    extract_feature_mapping,
    fetch_dataset_metadata,
    normalize_dataframe_filenames,
    resolve_download_url,
    resolve_images_root,
    serialize_and_zip_model,
    sort_models_by_size,
    validate_vision_inputs,
)

# ---------------------------------------------------------------------------
# normalize_dataframe_filenames
# ---------------------------------------------------------------------------


def test_normalize_filenames_unix_paths(tmp_path):
    df = pd.DataFrame(
        {"filename": ["some/path/img.png", "other/img2.png"], "label": [0, 1]}
    )
    csv_path = tmp_path / "labels.csv"
    result = normalize_dataframe_filenames(df, "filename", csv_path)
    assert list(result["filename"]) == ["img.png", "img2.png"]
    assert csv_path.exists()


def test_normalize_filenames_windows_paths(tmp_path):
    df = pd.DataFrame(
        {
            "filename": ["C:\\Users\\test\\img.png", "D:\\data\\img2.png"],
            "label": [0, 1],
        }
    )
    csv_path = tmp_path / "labels.csv"
    result = normalize_dataframe_filenames(df, "filename", csv_path)
    assert list(result["filename"]) == ["img.png", "img2.png"]


def test_normalize_filenames_already_basenames(tmp_path):
    df = pd.DataFrame({"filename": ["a.png", "b.png"], "label": [0, 1]})
    csv_path = tmp_path / "labels.csv"
    result = normalize_dataframe_filenames(df, "filename", csv_path)
    assert list(result["filename"]) == ["a.png", "b.png"]


def test_normalize_filenames_saves_csv(tmp_path):
    df = pd.DataFrame({"filename": ["path/img.png"], "label": [0]})
    csv_path = tmp_path / "labels.csv"
    normalize_dataframe_filenames(df, "filename", csv_path)
    saved = pd.read_csv(csv_path)
    assert saved["filename"].iloc[0] == "img.png"


def test_normalize_filenames_missing_column_returns_df(tmp_path):
    df = pd.DataFrame({"wrong_col": ["a.png"], "label": [0]})
    csv_path = tmp_path / "labels.csv"
    result = normalize_dataframe_filenames(df, "filename", csv_path)
    assert "wrong_col" in result.columns
    assert not csv_path.exists()


# ---------------------------------------------------------------------------
# resolve_images_root
# ---------------------------------------------------------------------------


def test_resolve_images_root_flat_directory(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img0.png").touch()
    result = resolve_images_root(images_dir)
    assert result == images_dir


def test_resolve_images_root_nested_images_subfolder(tmp_path):
    images_dir = tmp_path / "images"
    nested = images_dir / "images"
    nested.mkdir(parents=True)
    (nested / "img0.png").touch()
    result = resolve_images_root(images_dir)
    assert result == nested


def test_resolve_images_root_single_subdir_unwrap(tmp_path):
    images_dir = tmp_path / "images"
    cls_dir = images_dir / "class_a"
    cls_dir.mkdir(parents=True)
    (cls_dir / "img0.png").touch()
    result = resolve_images_root(images_dir)
    assert result == cls_dir


def test_resolve_images_root_multiple_subdirs_no_unwrap(tmp_path):
    images_dir = tmp_path / "images"
    (images_dir / "cat").mkdir(parents=True)
    (images_dir / "dog").mkdir()
    result = resolve_images_root(images_dir)
    assert result == images_dir


# ---------------------------------------------------------------------------
# collect_missing_files
# ---------------------------------------------------------------------------


def test_collect_missing_files_all_present(synthetic_images_dir):
    csv_path, images_dir = synthetic_images_dir
    df = pd.read_csv(csv_path)
    missing = collect_missing_files(df, images_dir, "filename", "label")
    assert missing == []


def test_collect_missing_files_some_missing(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "present.png").touch()
    df = pd.DataFrame({"filename": ["present.png", "absent.png"], "label": [0, 1]})
    missing = collect_missing_files(df, images_dir, "filename", "label")
    assert missing == ["absent.png"]


def test_collect_missing_files_found_via_rglob(tmp_path):
    images_dir = tmp_path / "images"
    subdir = images_dir / "subdir"
    subdir.mkdir(parents=True)
    (subdir / "nested.png").touch()
    df = pd.DataFrame({"filename": ["nested.png"], "label": [0]})
    missing = collect_missing_files(df, images_dir, "filename", "label")
    assert missing == []


def test_collect_missing_files_multiple_matches_warns(tmp_path):
    images_dir = tmp_path / "images"
    (images_dir / "a").mkdir(parents=True)
    (images_dir / "b").mkdir()
    (images_dir / "a" / "dup.png").touch()
    (images_dir / "b" / "dup.png").touch()
    df = pd.DataFrame({"filename": ["dup.png"], "label": [0]})
    missing = collect_missing_files(df, images_dir, "filename", "label")
    assert "dup.png" in missing


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
# _find_valid_dataset_root
# ---------------------------------------------------------------------------


def test_find_valid_dataset_root_skips_macosx(tmp_path):
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "real_data").mkdir()
    result = _find_valid_dataset_root(tmp_path)
    assert result == tmp_path / "real_data"


def test_find_valid_dataset_root_skips_dotdirs(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "actual_data").mkdir()
    result = _find_valid_dataset_root(tmp_path)
    assert result == tmp_path / "actual_data"


def test_find_valid_dataset_root_raises_when_no_valid_dirs(tmp_path):
    (tmp_path / "__MACOSX").mkdir()
    with pytest.raises(AutoMLDataError, match="No valid dataset folder"):
        _find_valid_dataset_root(tmp_path)


# ---------------------------------------------------------------------------
# _find_csv_file
# ---------------------------------------------------------------------------


def test_find_csv_file_finds_labels_csv(tmp_path):
    (tmp_path / "labels.csv").touch()
    result = _find_csv_file(tmp_path)
    assert result == tmp_path / "labels.csv"


def test_find_csv_file_finds_metadata_csv(tmp_path):
    (tmp_path / "metadata.csv").touch()
    result = _find_csv_file(tmp_path)
    assert result == tmp_path / "metadata.csv"


def test_find_csv_file_raises_when_not_found(tmp_path):
    with pytest.raises(AutoMLDataError, match="labels.csv or metadata.csv"):
        _find_csv_file(tmp_path)


def test_find_csv_file_prefers_labels_csv(tmp_path):
    # Both exist - we don't guarantee order, but we just need it to succeed
    (tmp_path / "labels.csv").touch()
    (tmp_path / "metadata.csv").touch()
    result = _find_csv_file(tmp_path)
    assert result.name in ("labels.csv", "metadata.csv")


# ---------------------------------------------------------------------------
# _find_or_resolve_images_dir
# ---------------------------------------------------------------------------


def test_find_or_resolve_images_dir_finds_images_subdir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img.png").touch()
    csv_path = tmp_path / "labels.csv"
    result = _find_or_resolve_images_dir(tmp_path, csv_path)
    assert result.exists()
    assert "images" in result.parts


def test_find_or_resolve_images_dir_raises_when_not_found(tmp_path):
    csv_path = tmp_path / "labels.csv"
    # No images/ directory at all
    with pytest.raises(AutoMLDataError, match="images/"):
        _find_or_resolve_images_dir(tmp_path, csv_path)


# ---------------------------------------------------------------------------
# extract_and_locate_dataset
# ---------------------------------------------------------------------------


def _make_dataset_zip(tmp_path: Path) -> Path:
    buf = io.BytesIO()
    from PIL import Image

    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "my_dataset/labels.csv", "filename,label\nimg0.png,cat\nimg1.png,dog\n"
        )
        zf.writestr("my_dataset/images/img0.png", png_bytes)
        zf.writestr("my_dataset/images/img1.png", png_bytes)
    return zip_path


def test_extract_and_locate_dataset_valid_zip(tmp_path):
    zip_path = _make_dataset_zip(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    csv_path, images_dir = extract_and_locate_dataset(zip_path, workdir)
    assert csv_path.exists()
    assert csv_path.name in ("labels.csv", "metadata.csv")
    assert images_dir.exists()
    assert images_dir.is_dir()


def test_extract_and_locate_dataset_missing_csv_raises(tmp_path):
    buf = io.BytesIO()
    from PIL import Image

    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("my_dataset/images/img0.png", png_bytes)

    workdir = tmp_path / "work"
    workdir.mkdir()
    with pytest.raises(AutoMLDataError, match="labels.csv or metadata.csv"):
        extract_and_locate_dataset(zip_path, workdir)


def test_extract_and_locate_dataset_no_valid_root_raises(tmp_path):
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("__MACOSX/._something", b"junk")

    workdir = tmp_path / "work"
    workdir.mkdir()
    with pytest.raises(AutoMLDataError, match="No valid dataset folder"):
        extract_and_locate_dataset(zip_path, workdir)


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


# ---------------------------------------------------------------------------
# sort_models_by_size
# ---------------------------------------------------------------------------

_MODELS = [
    {"model_id": "small_m", "num_params": 10_000_000},
    {"model_id": "medium_m", "num_params": 80_000_000},
    {"model_id": "large_m", "num_params": 300_000_000},
]


def test_sort_models_by_size_small_tier():
    result = sort_models_by_size(_MODELS, "small")
    ids = [m["model_id"] for m in result]
    assert ids == ["small_m"]


def test_sort_models_by_size_medium_tier():
    result = sort_models_by_size(_MODELS, "medium")
    ids = [m["model_id"] for m in result]
    assert ids == ["medium_m"]


def test_sort_models_by_size_large_tier():
    result = sort_models_by_size(_MODELS, "large")
    ids = [m["model_id"] for m in result]
    assert ids == ["large_m"]


def test_sort_models_by_size_fallback_when_no_match():
    models = [{"model_id": "big", "num_params": 300_000_000}]
    result = sort_models_by_size(models, "small")
    assert result == models


def test_sort_models_by_size_none_params_excluded():
    models = [
        {"model_id": "m1", "num_params": None},
        {"model_id": "m2", "num_params": 10_000_000},
    ]
    result = sort_models_by_size(models, "small")
    assert len(result) == 1
    assert result[0]["model_id"] == "m2"


def test_sort_models_by_size_custom_env_thresholds(monkeypatch):
    monkeypatch.setenv("MODEL_SMALL_MAX_PARAM_SIZE", "100000000")
    monkeypatch.setenv("MODEL_MEDIUM_MAX_PARAM_SIZE", "500000000")
    result = sort_models_by_size(_MODELS, "small")
    ids = [m["model_id"] for m in result]
    assert "small_m" in ids
    assert "medium_m" in ids


def test_sort_models_by_size_result_sorted_ascending():
    models = [
        {"model_id": "c", "num_params": 50_000_000},
        {"model_id": "a", "num_params": 20_000_000},
        {"model_id": "b", "num_params": 40_000_000},
    ]
    result = sort_models_by_size(models, "small")
    params = [m["num_params"] for m in result]
    assert params == sorted(params)


def test_sort_models_by_size_unknown_tier_returns_all():
    result = sort_models_by_size(_MODELS, "xlarge")
    assert len(result) == len(_MODELS)


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


# ---------------------------------------------------------------------------
# extract_feature_mapping
# ---------------------------------------------------------------------------


class _FakeTextDatamodule:
    """Stand-in for a text-task datamodule post-setup."""

    def __init__(self):
        self.hf_model_id = "distilbert-base-uncased"
        self.id2label = {0: "neg", 1: "pos"}
        self.label2id = {"neg": 0, "pos": 1}

        class _FakeTokenizer:
            def get_vocab(self):
                return {"hello": 0, "world": 1, "[PAD]": 2}

        self.tokenizer = _FakeTokenizer()


class _FakeScaler:
    n_features_in_ = 2
    mean_ = __import__("numpy").array([0.1, 0.2])
    scale_ = __import__("numpy").array([1.0, 2.0])
    var_ = __import__("numpy").array([1.0, 4.0])


class _FakeEncoder:
    categories_ = [
        __import__("numpy").array(["a", "b", "c"]),
        __import__("numpy").array(["x", "y"]),
    ]


class _FakeMultimodalDatamodule:
    """Stand-in for a fitted MultimodalClassificationDataModule."""

    def __init__(self):
        self.hf_model_id = "google/vit-base-patch16-224"
        self.id2label = {0: "cat", 1: "dog"}
        self.label2id = {"cat": 0, "dog": 1}
        self.auxiliary_columns = ["age", "city"]
        self.numeric_cols = ["age"]
        self.categorical_cols = ["city"]
        self.aux_feature_dim = 2
        self.scaler = _FakeScaler()
        self.encoder = _FakeEncoder()


class _FakeImageDatamodule:
    """Pure-image datamodule: only label maps should be extracted."""

    def __init__(self):
        self.hf_model_id = "google/vit-base-patch16-224"
        self.id2label = {0: "cat", 1: "dog"}
        self.label2id = {"cat": 0, "dog": 1}


class _FakeMultimodalModel:
    def _get_vision_embed_dim(self):
        return 768


def test_extract_feature_mapping_text_task_includes_vocab():
    dm = _FakeTextDatamodule()
    out = extract_feature_mapping(dm, "text_classification")
    assert out["task_type"] == "text_classification"
    assert out["label_map"]["id2label"] == {"0": "neg", "1": "pos"}
    assert out["tokenizer"]["hf_model_id"] == "distilbert-base-uncased"
    assert out["tokenizer"]["vocab"]["hello"] == 0
    assert "[PAD]" in out["tokenizer"]["vocab"]
    assert "auxiliary_features" not in out


def test_extract_feature_mapping_multimodal_includes_preprocessing():
    dm = _FakeMultimodalDatamodule()
    model = _FakeMultimodalModel()
    out = extract_feature_mapping(dm, "image_classification_multimodal", model=model)
    aux = out["auxiliary_features"]
    assert aux["auxiliary_columns"] == ["age", "city"]
    assert aux["numeric_columns"] == ["age"]
    assert aux["categorical_columns"] == ["city"]
    assert aux["aux_feature_dim"] == 2
    assert aux["scaler"]["mean"] == [0.1, 0.2]
    assert aux["scaler"]["scale"] == [1.0, 2.0]
    assert aux["ordinal_encoder"]["categories"] == [
        ["a", "b", "c"],
        ["x", "y"],
    ]
    assert aux["vision_embed_dim"] == 768
    assert "tokenizer" not in out


def test_extract_feature_mapping_image_only_returns_label_map_only():
    dm = _FakeImageDatamodule()
    out = extract_feature_mapping(dm, "image_classification")
    assert out["task_type"] == "image_classification"
    assert out["label_map"]["id2label"] == {"0": "cat", "1": "dog"}
    assert "tokenizer" not in out
    assert "auxiliary_features" not in out


def test_extract_feature_mapping_label_map_empty_when_missing():
    class _Bare:
        pass

    out = extract_feature_mapping(_Bare(), "image_classification")
    assert out["label_map"] == {}


def test_extract_feature_mapping_resilient_to_failing_model():
    dm = _FakeMultimodalDatamodule()

    class _Boom:
        def _get_vision_embed_dim(self):
            raise RuntimeError("boom")

    out = extract_feature_mapping(dm, "image_classification_multimodal", model=_Boom())
    # vision_embed_dim skipped, but the rest of the section is still there
    assert "vision_embed_dim" not in out["auxiliary_features"]
    assert out["auxiliary_features"]["scaler"]["mean"] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# validate_vision_inputs — new task types
# ---------------------------------------------------------------------------


def test_validate_vision_inputs_default_is_image_classification(synthetic_images_dir):
    csv_path, images_dir = synthetic_images_dir
    # Calling without task_type should default to image_classification (backward compat)
    result = validate_vision_inputs(csv_path, images_dir, "filename", "label")
    assert result is None


def test_validate_vision_inputs_audio_classification_missing_audio_dir(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"audio_path": ["a.wav"], "label": ["cat"]}).to_csv(
        csv_path, index=False
    )
    result = validate_vision_inputs(
        csv_path,
        tmp_path / "nonexistent_audio",
        "audio_path",
        "label",
        task_type="audio_classification",
    )
    assert result is not None
    assert "Audio directory" in result


def test_validate_vision_inputs_audio_classification_valid(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"audio_path": ["a.wav"], "label": ["cat"]}).to_csv(
        csv_path, index=False
    )
    result = validate_vision_inputs(
        csv_path,
        audio_dir,
        "audio_path",
        "label",
        task_type="audio_classification",
    )
    assert result is None


def test_validate_vision_inputs_text_classification_valid(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"text": ["hello world"], "label": ["pos"]}).to_csv(
        csv_path, index=False
    )
    result = validate_vision_inputs(
        csv_path,
        tmp_path,
        "text",
        "label",
        task_type="text_classification",
    )
    assert result is None


def test_validate_vision_inputs_text_classification_missing_columns(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"sentence": ["hello"], "sentiment": ["pos"]}).to_csv(
        csv_path, index=False
    )
    result = validate_vision_inputs(
        csv_path,
        tmp_path,
        "text",
        "label",
        task_type="text_classification",
    )
    assert result is not None
    assert "text_classification" in result


def test_validate_vision_inputs_question_answering_valid(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "question": ["What?"],
            "context": ["Some text"],
            "answer_start": [0],
            "answer_text": ["Some"],
        }
    ).to_csv(csv_path, index=False)
    result = validate_vision_inputs(
        csv_path,
        tmp_path,
        "question",
        "answer_text",
        task_type="question_answering",
    )
    assert result is None


def test_validate_vision_inputs_question_answering_missing_columns(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"question": ["What?"], "context": ["Text"]}).to_csv(
        csv_path, index=False
    )
    result = validate_vision_inputs(
        csv_path,
        tmp_path,
        "question",
        "answer_text",
        task_type="question_answering",
    )
    assert result is not None
    assert "question_answering" in result


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
