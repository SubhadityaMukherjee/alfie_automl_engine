"""Integration tests for the text AutoML router."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)

_TEXT_PARAMS = {
    "user_id": "user1",
    "dataset_id": "ds1",
    "text_column": "text",
    "label_column": "label",
    "task_type": "text_classification",
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
    "app.text_automl.router.deployment_instructions",
    return_value="deploy instructions",
)
def test_deployment_instructions(mock_fn):
    resp = client.post("/automl/text/deployment_instructions/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


@patch(
    "app.text_automl.router.text_data_instructions",
    return_value="data instructions",
)
def test_accepted_format(mock_fn):
    resp = client.post("/automl/text/accepted_format/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


# ---------------------------------------------------------------------------
# find_best_model_for_text — metadata errors
# ---------------------------------------------------------------------------


@patch("app.text_automl.orchestrator.fetch_dataset_metadata")
def test_best_model_metadata_error_returns_500(mock_fetch):
    mock_fetch.side_effect = RuntimeError("metadata crash")
    resp = client.post("/automl/text/best_model/", data=_TEXT_PARAMS)
    assert resp.status_code == 500


@patch("app.text_automl.orchestrator.fetch_dataset_metadata")
def test_best_model_non_zip_returns_400(mock_fetch):
    mock_fetch.return_value = _make_metadata(file_type="csv")
    resp = client.post("/automl/text/best_model/", data=_TEXT_PARAMS)
    assert resp.status_code == 400
    assert "ZIP" in resp.json()["error"]


# ---------------------------------------------------------------------------
# find_best_model_for_text — validation errors
# ---------------------------------------------------------------------------


@patch("app.text_automl.orchestrator.validate_text_inputs")
@patch("app.text_automl.orchestrator.extract_and_locate_dataset")
@patch("app.text_automl.orchestrator.download_dataset")
@patch(
    "app.text_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.text_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_validation_error(
    mock_fetch, mock_resolve, mock_download, mock_extract, mock_validate
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/media"))
    mock_validate.return_value = (
        "Required column(s) missing for text_classification: ['text']"
    )

    resp = client.post("/automl/text/best_model/", data=_TEXT_PARAMS)
    assert resp.status_code == 400
    assert "text_classification" in resp.json()["error"]


@patch("app.text_automl.orchestrator.validate_text_inputs", return_value=None)
@patch("app.text_automl.orchestrator.extract_and_locate_dataset")
@patch("app.text_automl.orchestrator.download_dataset")
@patch(
    "app.text_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.text_automl.orchestrator.fetch_dataset_metadata",
    return_value=_make_metadata(),
)
def test_best_model_unsupported_task_type(
    mock_fetch, mock_resolve, mock_download, mock_extract, mock_validate
):
    from pathlib import Path

    mock_download.return_value = Path("/tmp/dataset.zip")
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/media"))

    params = {**_TEXT_PARAMS, "task_type": "image_classification"}
    resp = client.post("/automl/text/best_model/", data=params)
    assert resp.status_code == 400
    assert "Unsupported task_type" in resp.json()["error"]


# ---------------------------------------------------------------------------
# find_best_model_for_text — happy path
# ---------------------------------------------------------------------------


@patch("app.text_automl.orchestrator.upload_model")
@patch(
    "app.text_automl.orchestrator.build_upload_payload",
    return_value=("model_id", {"key": "val"}),
)
@patch(
    "app.text_automl.orchestrator.convert_leaderboard_safely",
    return_value=({"score": 0.95}, "lb_str"),
)
@patch("app.text_automl.orchestrator.serialize_and_zip_model")
@patch("app.text_automl.orchestrator.train_automl", return_value=MagicMock())
@patch("app.text_automl.orchestrator.validate_text_inputs", return_value=None)
@patch("app.text_automl.orchestrator.extract_and_locate_dataset")
@patch("app.text_automl.orchestrator.download_dataset")
@patch(
    "app.text_automl.orchestrator.resolve_download_url",
    return_value="http://download",
)
@patch(
    "app.text_automl.orchestrator.fetch_dataset_metadata",
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
    mock_extract.return_value = (Path("/tmp/labels.csv"), Path("/tmp/media"))

    zip_path = MagicMock()
    zip_path.exists.return_value = True
    mock_serialize.return_value = zip_path

    upload_resp = MagicMock()
    upload_resp.status_code = 200
    mock_upload.return_value = upload_resp

    resp = client.post("/automl/text/best_model/", data=_TEXT_PARAMS)
    assert resp.status_code == 200
    assert "Text AutoML training completed" in resp.json()["message"]
    assert resp.json()["leaderboard"] == "lb_str"
