"""Integration tests for the tabular AutoML router."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.tabular_automl.router import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Static endpoints
# ---------------------------------------------------------------------------


@patch(
    "app.tabular_automl.router.deployment_instructions",
    return_value="deploy instructions",
)
def test_deployment_instructions(mock_fn):
    resp = client.post("/automl_tabular/deployment_instructions/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


@patch(
    "app.tabular_automl.router.tabular_data_instructions",
    return_value="data instructions",
)
def test_accepted_format(mock_fn):
    resp = client.post("/automl_tabular/accepted_format/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


# ---------------------------------------------------------------------------
# best_model input validation
# ---------------------------------------------------------------------------


def test_best_model_missing_user_id():
    resp = client.post(
        "/automl_tabular/best_model/",
        data={
            "user_id": "",
            "dataset_id": "ds1",
            "target_column_name": "target",
            "task_type": "classification",
            "time_budget": 10,
        },
    )
    assert resp.status_code == 400
    assert "user_id" in resp.json()["error"]


def test_best_model_missing_dataset_id():
    resp = client.post(
        "/automl_tabular/best_model/",
        data={
            "user_id": "user1",
            "dataset_id": "",
            "target_column_name": "target",
            "task_type": "classification",
            "time_budget": 10,
        },
    )
    assert resp.status_code == 400
    assert "dataset_id" in resp.json()["error"]


def test_best_model_missing_target_column():
    resp = client.post(
        "/automl_tabular/best_model/",
        data={
            "user_id": "user1",
            "dataset_id": "ds1",
            "target_column_name": "",
            "task_type": "classification",
            "time_budget": 10,
        },
    )
    assert resp.status_code == 400
    assert "target_column_name" in resp.json()["error"]


def test_best_model_invalid_task_type():
    resp = client.post(
        "/automl_tabular/best_model/",
        data={
            "user_id": "user1",
            "dataset_id": "ds1",
            "target_column_name": "target",
            "task_type": "invalid_task",
            "time_budget": 10,
        },
    )
    assert resp.status_code == 400
    assert "task_type" in resp.json()["error"]


def test_best_model_invalid_time_budget():
    resp = client.post(
        "/automl_tabular/best_model/",
        data={
            "user_id": "user1",
            "dataset_id": "ds1",
            "target_column_name": "target",
            "task_type": "tabular_classification",
            "time_budget": 0,
        },
    )
    assert resp.status_code == 400
    assert "time_budget" in resp.json()["error"]


def test_best_model_invalid_dataset_split():
    resp = client.post(
        "/automl_tabular/best_model/",
        data={
            "user_id": "user1",
            "dataset_id": "ds1",
            "target_column_name": "target",
            "task_type": "tabular_classification",
            "time_budget": 10,
            "dataset_split": "invalid",
        },
    )
    assert resp.status_code == 400
    assert "dataset_split" in resp.json()["error"]
