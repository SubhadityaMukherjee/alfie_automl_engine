"""Shared fixtures for vision_automl tests."""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dataset fixtures shared with the ML-engine dataset tests live in
# tests/conftest.py (synthetic_images_dir, class_structured_images_dir).
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_optuna_result():
    return {
        "best_value": 0.123,
        "best_params": {"lr": 0.001, "model_id": "google/efficientnet-b0"},
        "best_metrics": {"loss": 0.123, "accuracy": 0.87, "f1_macro": 0.85},
        "n_trials": 3,
        "model_dir": Path("/tmp/trial_0"),
    }


@pytest.fixture
def fake_metadata():
    return {
        "file_type": "zip",
        "original_filename": "dataset.zip",
        "version": "v1",
        "custom_metadata": {},
    }


@pytest.fixture
def fake_metadata_with_splits():
    return {
        "file_type": "zip",
        "original_filename": "dataset.zip",
        "version": "v1",
        "custom_metadata": {"split": True},
    }
