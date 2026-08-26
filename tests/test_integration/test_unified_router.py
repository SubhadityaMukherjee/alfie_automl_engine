"""Integration tests for the unified router mounted on the combined app."""

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _api_paths() -> set[str]:
    return {route.path for route in app.routes if isinstance(route, APIRoute)}


def test_health_and_ready_mounted_at_root():
    paths = _api_paths()
    assert "/health" in paths
    assert "/ready" in paths


def test_unified_router_exposes_all_service_endpoints():
    paths = _api_paths()
    assert paths >= {
        # Tabular AutoML
        "/automl/tabular/deployment_instructions/",
        "/automl/tabular/accepted_format/",
        "/automl/tabular/best_model/",
        # Vision AutoML
        "/automl/vision/deployment_instructions/",
        "/automl/vision/accepted_format/",
        "/automl/vision/best_model/",
        "/automl/vision/multimodal_best_model/",
        # Audio AutoML
        "/automl/audio/deployment_instructions/",
        "/automl/audio/accepted_format/",
        "/automl/audio/best_model/",
        # Text AutoML
        "/automl/text/deployment_instructions/",
        "/automl/text/accepted_format/",
        "/automl/text/best_model/",
        # AutoML+
        "/automl/automl_plus/accepted_format/",
        "/automl/automl_plus/image_tools/image_to_website/",
        "/automl/automl_plus/image_tools/run_on_image/",
        "/automl/automl_plus/image_tools/run_on_image_stream/",
        "/automl/automl_plus/web_access/check-alt-text/",
        "/automl/automl_plus/web_access/analyze/",
    }


def test_legacy_prefixes_are_gone():
    paths = _api_paths()
    assert not any(
        path.startswith(("/automl_tabular", "/automl_vision", "/automlplus"))
        for path in paths
    )


def test_endpoints_listing_covers_every_route():
    resp = client.get("/automl/endpoints")
    assert resp.status_code == 200
    listed = {e["path"] for e in resp.json()}
    assert listed == _api_paths()


def test_endpoints_listing_is_llm_readable():
    resp = client.get("/automl/endpoints")
    entries = resp.json()
    assert len(entries) > 0

    by_path = {e["path"]: e for e in entries}
    vision = by_path["/automl/vision/best_model/"]
    assert vision["methods"] == ["POST"]
    assert "vision" in vision["tags"]
    assert vision["summary"]
    assert "Fetch a vision dataset" in vision["description"]

    health = by_path["/health"]
    assert health["methods"] == ["GET"]

    # Sorted by path for stable, diff-friendly output.
    paths = [e["path"] for e in entries]
    assert paths == sorted(paths)
