"""Integration tests for the vision AutoML router."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.vision_automl.router import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


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
