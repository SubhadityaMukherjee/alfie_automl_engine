"""Integration tests for the unified router mounted on the combined app."""

from fastapi.routing import APIRoute

from app.main import app


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
