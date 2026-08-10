"""Integration tests for the vision AutoML router."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.vision_automl.router import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)

_VISION_PARAMS = {
    "user_id": "user1",
    "dataset_id": "ds1",
    "filename_column": "filename",
    "label_column": "label",
    "task_type": "image_classification",
    "time_budget": 60,
    "model_size": "small",
}

_MULTIMODAL_PARAMS = {
    "user_id": "user1",
    "dataset_id": "ds1",
    "filename_column": "filename",
    "label_column": "label",
    "time_budget": 60,
    "model_size": "small",
}


def _make_metadata(file_type="zip"):
    return {
        "file_type": file_type,
        "original_filename": "dataset.zip",
    }


# ---------------------------------------------------------------------------
# Static endpoints
# ---------------------------------------------------------------------------


@patch(
    "app.vision_automl.router.deployment_instructions",
    return_value="deploy instructions",
)
def test_deployment_instructions(mock_fn):
    resp = client.post("/automl_vision/deployment_instructions/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


@patch(
    "app.vision_automl.router.vision_data_instructions",
    return_value="data instructions",
)
def test_accepted_format(mock_fn):
    resp = client.post("/automl_vision/accepted_format/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


# ---------------------------------------------------------------------------
# find_best_model_for_vision — metadata errors
# ---------------------------------------------------------------------------


@patch("app.vision_automl.orchestrator.fetch_dataset_metadata")
def test_best_model_metadata_error_returns_500(mock_fetch):
    mock_fetch.side_effect = RuntimeError("metadata crash")
    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 500


@patch("app.vision_automl.orchestrator.fetch_dataset_metadata")
def test_best_model_non_zip_returns_400(mock_fetch):
    mock_fetch.return_value = _make_metadata(file_type="csv")
    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 400
    assert "ZIP" in resp.json()["error"]


# ---------------------------------------------------------------------------
# find_best_model_for_vision — download & extract errors
# ---------------------------------------------------------------------------


@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_download_error(mock_fetch, mock_resolve, mock_download):
    mock_download.side_effect = RuntimeError("download failed")
    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 500


@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_extract_error(
    mock_fetch, mock_resolve, mock_download, mock_extract
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.side_effect = RuntimeError("extraction failed")
    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# find_best_model_for_vision — validation errors
# ---------------------------------------------------------------------------


@patch("app.vision_automl.orchestrator.validate_vision_inputs")
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_validation_error(
    mock_fetch, mock_resolve, mock_download, mock_extract, mock_validate
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/images"))
    mock_validate.return_value = "Missing 5 image file(s)"

    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 400
    assert "Missing" in resp.json()["error"]


@patch("app.vision_automl.orchestrator.validate_vision_inputs", return_value=None)
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_unsupported_task_type(
    mock_fetch, mock_resolve, mock_download, mock_extract, mock_validate
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/images"))

    params = {**_VISION_PARAMS, "task_type": "unsupported_task"}
    resp = client.post("/automl_vision/best_model/", data=params)
    assert resp.status_code == 400
    assert "Unsupported task_type" in resp.json()["error"]


# ---------------------------------------------------------------------------
# find_best_model_for_vision — training errors
# ---------------------------------------------------------------------------


@patch("app.vision_automl.orchestrator.train_automl")
@patch("app.vision_automl.orchestrator.validate_vision_inputs", return_value=None)
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_training_validation_error(
    mock_fetch, mock_resolve, mock_download, mock_extract, mock_validate, mock_train
):
    from pathlib import Path

    from app.core.exceptions import AutoMLValidationError

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/images"))
    mock_train.side_effect = AutoMLValidationError("bad data")

    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 400


@patch("app.vision_automl.orchestrator.train_automl")
@patch("app.vision_automl.orchestrator.validate_vision_inputs", return_value=None)
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_training_runtime_error(
    mock_fetch, mock_resolve, mock_download, mock_extract, mock_validate, mock_train
):
    from pathlib import Path

    from app.core.exceptions import AutoMLRuntimeError

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/images"))
    mock_train.side_effect = AutoMLRuntimeError("training crashed")

    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# find_best_model_for_vision — upload errors
# ---------------------------------------------------------------------------


@patch("app.vision_automl.orchestrator.upload_model")
@patch(
    "app.vision_automl.orchestrator.build_upload_payload",
    return_value=("model_id", {"key": "val"}),
)
@patch(
    "app.vision_automl.orchestrator.convert_leaderboard_safely",
    return_value=({"score": 0.9}, "lb_str"),
)
@patch("app.vision_automl.orchestrator.serialize_and_zip_model")
@patch("app.vision_automl.orchestrator.train_automl", return_value=MagicMock())
@patch("app.vision_automl.orchestrator.validate_vision_inputs", return_value=None)
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_upload_failure(
    mock_fetch,
    mock_resolve,
    mock_download,
    mock_extract,
    mock_validate,
    mock_train,
    mock_serialize,
    mock_convert,
    mock_payload,
    mock_upload,
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/images"))

    zip_path = MagicMock()
    zip_path.exists.return_value = True
    mock_serialize.return_value = zip_path

    upload_resp = MagicMock()
    upload_resp.status_code = 503
    upload_resp.text = "Service unavailable"
    mock_upload.return_value = upload_resp

    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 503
    assert "upload" in resp.json()["error"].lower()


# ---------------------------------------------------------------------------
# find_best_model_for_vision — happy path
# ---------------------------------------------------------------------------


@patch("app.vision_automl.orchestrator.upload_model")
@patch(
    "app.vision_automl.orchestrator.build_upload_payload",
    return_value=("model_id", {"key": "val"}),
)
@patch(
    "app.vision_automl.orchestrator.convert_leaderboard_safely",
    return_value=({"score": 0.95}, "lb_str"),
)
@patch("app.vision_automl.orchestrator.serialize_and_zip_model")
@patch("app.vision_automl.orchestrator.train_automl", return_value=MagicMock())
@patch("app.vision_automl.orchestrator.validate_vision_inputs", return_value=None)
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_success(
    mock_fetch,
    mock_resolve,
    mock_download,
    mock_extract,
    mock_validate,
    mock_train,
    mock_serialize,
    mock_convert,
    mock_payload,
    mock_upload,
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/images"))

    zip_path = MagicMock()
    zip_path.exists.return_value = True
    mock_serialize.return_value = zip_path

    upload_resp = MagicMock()
    upload_resp.status_code = 200
    mock_upload.return_value = upload_resp

    resp = client.post("/automl_vision/best_model/", data=_VISION_PARAMS)
    assert resp.status_code == 200
    assert "Vision AutoML training completed" in resp.json()["message"]
    assert resp.json()["leaderboard"] == "lb_str"


# ---------------------------------------------------------------------------
# find_best_model_for_multimodal_vision — validation errors
# ---------------------------------------------------------------------------


@patch("app.vision_automl.orchestrator.validate_multimodal_inputs")
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_multimodal_validation_error(
    mock_fetch, mock_resolve, mock_download, mock_extract, mock_validate
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/images"))
    mock_validate.return_value = ("No auxiliary columns found", [])

    resp = client.post("/automl_vision/multimodal_best_model/", data=_MULTIMODAL_PARAMS)
    assert resp.status_code == 400
    assert "auxiliary" in resp.json()["error"].lower()


@patch("app.vision_automl.orchestrator.validate_multimodal_inputs")
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_multimodal_non_zip_returns_400(
    mock_fetch, mock_resolve, mock_download, mock_extract, mock_validate
):
    mock_fetch.return_value = _make_metadata(file_type="csv")
    resp = client.post("/automl_vision/multimodal_best_model/", data=_MULTIMODAL_PARAMS)
    assert resp.status_code == 400
    assert "ZIP" in resp.json()["error"]


# ---------------------------------------------------------------------------
# find_best_model_for_multimodal_vision — happy path
# ---------------------------------------------------------------------------


@patch("app.vision_automl.orchestrator.upload_model")
@patch(
    "app.vision_automl.orchestrator.build_upload_payload",
    return_value=("model_id", {"key": "val"}),
)
@patch(
    "app.vision_automl.orchestrator.convert_leaderboard_safely",
    return_value=({"score": 0.88}, "lb_str"),
)
@patch("app.vision_automl.orchestrator.serialize_and_zip_model")
@patch(
    "app.vision_automl.orchestrator.train_automl_multimodal", return_value=MagicMock()
)
@patch(
    "app.vision_automl.orchestrator.validate_multimodal_inputs",
    return_value=(None, ["feature1", "feature2"]),
)
@patch("app.vision_automl.orchestrator.extract_and_locate_dataset")
@patch("app.vision_automl.orchestrator.download_dataset")
@patch(
    "app.vision_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.vision_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_multimodal_success(
    mock_fetch,
    mock_resolve,
    mock_download,
    mock_extract,
    mock_validate,
    mock_train,
    mock_serialize,
    mock_convert,
    mock_payload,
    mock_upload,
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/images"))

    zip_path = MagicMock()
    zip_path.exists.return_value = True
    mock_serialize.return_value = zip_path

    upload_resp = MagicMock()
    upload_resp.status_code = 200
    mock_upload.return_value = upload_resp

    resp = client.post("/automl_vision/multimodal_best_model/", data=_MULTIMODAL_PARAMS)
    assert resp.status_code == 200
    assert "Multimodal" in resp.json()["message"]
    assert resp.json()["auxiliary_columns"] == ["feature1", "feature2"]
