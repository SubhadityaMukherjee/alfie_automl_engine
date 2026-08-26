"""Integration tests for the tabular AutoML router."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)

_VALID_PARAMS = {
    "user_id": "user1",
    "dataset_id": "ds1",
    "target_column_name": "target",
    "task_type": "tabular_classification",
    "time_budget": 10,
}


def _make_download_side_effect():
    """Return a side_effect that writes a dummy file so dataset_path.exists() passes."""

    def _write_file(download_url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("col1,col2\n1,2\n")

    return _write_file


# ---------------------------------------------------------------------------
# Static endpoints
# ---------------------------------------------------------------------------


@patch(
    "app.tabular_automl.router.deployment_instructions",
    return_value="deploy instructions",
)
def test_deployment_instructions(mock_fn):
    resp = client.post("/automl/tabular/deployment_instructions/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


@patch(
    "app.tabular_automl.router.tabular_data_instructions",
    return_value="data instructions",
)
def test_accepted_format(mock_fn):
    resp = client.post("/automl/tabular/accepted_format/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


# ---------------------------------------------------------------------------
# best_model input validation (early guard checks)
# ---------------------------------------------------------------------------


def test_best_model_missing_user_id():
    params = {**_VALID_PARAMS, "user_id": ""}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "user_id" in resp.json()["error"]


def test_best_model_missing_dataset_id():
    params = {**_VALID_PARAMS, "dataset_id": ""}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "dataset_id" in resp.json()["error"]


def test_best_model_missing_target_column():
    params = {**_VALID_PARAMS, "target_column_name": ""}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "target_column_name" in resp.json()["error"]


def test_best_model_invalid_task_type():
    params = {**_VALID_PARAMS, "task_type": "invalid_task"}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "task_type" in resp.json()["error"]


def test_best_model_invalid_time_budget():
    params = {**_VALID_PARAMS, "time_budget": 0}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "time_budget" in resp.json()["error"]


def test_best_model_invalid_dataset_split():
    params = {**_VALID_PARAMS, "dataset_split": "invalid"}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "dataset_split" in resp.json()["error"]


def test_best_model_num_cpus_zero_rejected():
    """0 is not a valid CPU count — must be a positive integer."""
    params = {**_VALID_PARAMS, "num_cpus": 0}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "num_cpus" in resp.json()["error"]


def test_best_model_num_cpus_negative_rejected():
    params = {**_VALID_PARAMS, "num_cpus": -1}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "num_cpus" in resp.json()["error"]


def test_best_model_num_gpus_zero_rejected():
    """0 is not a valid GPU count — must be a positive integer."""
    params = {**_VALID_PARAMS, "num_gpus": 0}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "num_gpus" in resp.json()["error"]


def test_best_model_num_gpus_negative_rejected():
    params = {**_VALID_PARAMS, "num_gpus": -2}
    resp = client.post("/automl/tabular/best_model/", data=params)
    assert resp.status_code == 400
    assert "num_gpus" in resp.json()["error"]


# ---------------------------------------------------------------------------
# best_model — metadata fetching errors
# ---------------------------------------------------------------------------


@patch("app.tabular_automl.orchestrator.fetch_dataset_metadata")
def test_best_model_metadata_request_exception(mock_fetch):
    import requests

    mock_fetch.side_effect = requests.RequestException("connection failed")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 502
    assert "metadata" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.fetch_dataset_metadata")
def test_best_model_metadata_unexpected_error(mock_fetch):
    mock_fetch.side_effect = RuntimeError("unexpected")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 500
    assert "metadata" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.fetch_dataset_metadata", return_value={})
def test_best_model_metadata_empty_dict(mock_fetch):
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 502
    assert "Invalid or empty metadata" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.fetch_dataset_metadata", return_value=None)
def test_best_model_metadata_not_dict(mock_fetch):
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 502
    assert "Invalid or empty metadata" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.fetch_dataset_metadata")
def test_best_model_metadata_missing_file_type(mock_fetch):
    mock_fetch.return_value = {"something": "else"}
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 400
    assert "file_type" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.fetch_dataset_metadata")
def test_best_model_metadata_unsupported_file_type(mock_fetch):
    mock_fetch.return_value = {"file_type": "xyz"}
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["error"]


# ---------------------------------------------------------------------------
# best_model — download URL resolution errors
# ---------------------------------------------------------------------------


@patch("app.tabular_automl.orchestrator.resolve_download_url")
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv"},
)
def test_best_model_resolve_url_error(mock_fetch, mock_resolve):
    mock_resolve.side_effect = Exception("url resolution failed")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 500
    assert "download URL" in resp.json()["error"]


# ---------------------------------------------------------------------------
# best_model — dataset download errors
# ---------------------------------------------------------------------------


@patch("app.tabular_automl.orchestrator.download_dataset")
@patch(
    "app.tabular_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv", "original_filename": "train.csv"},
)
def test_best_model_download_request_exception(mock_fetch, mock_resolve, mock_download):
    import requests

    mock_download.side_effect = requests.RequestException("download failed")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 502
    assert "download" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.download_dataset")
@patch(
    "app.tabular_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv", "original_filename": "train.csv"},
)
def test_best_model_download_unexpected_error(mock_fetch, mock_resolve, mock_download):
    mock_download.side_effect = RuntimeError("unexpected download error")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 500
    assert "download" in resp.json()["error"]


# ---------------------------------------------------------------------------
# best_model — validation errors
# ---------------------------------------------------------------------------


@patch("app.tabular_automl.orchestrator.validate_tabular_inputs")
@patch(
    "app.tabular_automl.orchestrator.download_dataset",
    side_effect=_make_download_side_effect(),
)
@patch(
    "app.tabular_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv", "original_filename": "train.csv"},
)
def test_best_model_validation_returns_error(
    mock_fetch, mock_resolve, mock_download, mock_validate
):
    mock_validate.return_value = "Target column not found"
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 400
    assert "Target column not found" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.validate_tabular_inputs")
@patch(
    "app.tabular_automl.orchestrator.download_dataset",
    side_effect=_make_download_side_effect(),
)
@patch(
    "app.tabular_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv", "original_filename": "train.csv"},
)
def test_best_model_validation_exception(
    mock_fetch, mock_resolve, mock_download, mock_validate
):
    mock_validate.side_effect = Exception("validation crash")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 500
    assert "validation" in resp.json()["error"]


# ---------------------------------------------------------------------------
# best_model — training errors
# ---------------------------------------------------------------------------


@patch("app.tabular_automl.orchestrator.train_automl")
@patch("app.tabular_automl.orchestrator.validate_tabular_inputs", return_value=None)
@patch(
    "app.tabular_automl.orchestrator.download_dataset",
    side_effect=_make_download_side_effect(),
)
@patch(
    "app.tabular_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv", "original_filename": "train.csv"},
)
def test_best_model_training_validation_error(
    mock_fetch, mock_resolve, mock_download, mock_validate, mock_train
):
    from app.core.exceptions import AutoMLValidationError

    mock_train.side_effect = AutoMLValidationError("bad params")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 400
    assert "Training validation failed" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.train_automl")
@patch("app.tabular_automl.orchestrator.validate_tabular_inputs", return_value=None)
@patch(
    "app.tabular_automl.orchestrator.download_dataset",
    side_effect=_make_download_side_effect(),
)
@patch(
    "app.tabular_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv", "original_filename": "train.csv"},
)
def test_best_model_training_runtime_error(
    mock_fetch, mock_resolve, mock_download, mock_validate, mock_train
):
    from app.core.exceptions import AutoMLRuntimeError

    mock_train.side_effect = AutoMLRuntimeError("training crashed")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 500
    assert "Model training failed" in resp.json()["error"]


@patch("app.tabular_automl.orchestrator.train_automl")
@patch("app.tabular_automl.orchestrator.validate_tabular_inputs", return_value=None)
@patch(
    "app.tabular_automl.orchestrator.download_dataset",
    side_effect=_make_download_side_effect(),
)
@patch(
    "app.tabular_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv", "original_filename": "train.csv"},
)
def test_best_model_training_unexpected_error(
    mock_fetch, mock_resolve, mock_download, mock_validate, mock_train
):
    mock_train.side_effect = Exception("surprise")
    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 500
    assert "training" in resp.json()["error"]


# ---------------------------------------------------------------------------
# best_model — happy path
# ---------------------------------------------------------------------------


@patch("app.tabular_automl.orchestrator.upload_model")
@patch(
    "app.tabular_automl.orchestrator.build_upload_payload",
    return_value=("model_id", {"key": "val"}),
)
@patch(
    "app.tabular_automl.orchestrator.convert_leaderboard_safely",
    return_value=({"score": 0.9}, "leaderboard_str"),
)
@patch("app.tabular_automl.orchestrator.serialize_and_zip_predictor")
@patch(
    "app.tabular_automl.orchestrator.train_automl",
    return_value=(MagicMock(), MagicMock()),
)
@patch("app.tabular_automl.orchestrator.validate_tabular_inputs", return_value=None)
@patch(
    "app.tabular_automl.orchestrator.download_dataset",
    side_effect=_make_download_side_effect(),
)
@patch(
    "app.tabular_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.tabular_automl.orchestrator.fetch_dataset_metadata",
    return_value={"file_type": "csv", "original_filename": "train.csv"},
)
def test_best_model_success(
    mock_fetch,
    mock_resolve,
    mock_download,
    mock_validate,
    mock_train,
    mock_serialize,
    mock_convert,
    mock_payload,
    mock_upload,
):
    zip_path = MagicMock()
    zip_path.exists.return_value = True
    mock_serialize.return_value = zip_path

    mock_upload_response = MagicMock()
    mock_upload_response.status_code = 200
    mock_upload.return_value = mock_upload_response

    resp = client.post("/automl/tabular/best_model/", data=_VALID_PARAMS)
    assert resp.status_code == 200
    body = resp.json()
    assert "AutoML training completed" in body["message"]
    assert body["leaderboard"] == "leaderboard_str"
