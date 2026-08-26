"""Tests for app/ml_engine/model_search.py."""

from app.ml_engine.model_search import sort_models_by_size

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
