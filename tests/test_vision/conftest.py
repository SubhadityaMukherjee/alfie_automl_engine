"""Shared fixtures for vision_automl tests."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


def _make_tiny_png(path: Path) -> None:
    img = Image.new("RGB", (10, 10), color=(128, 64, 32))
    img.save(path)


@pytest.fixture
def synthetic_images_dir(tmp_path):
    """Flat images dir with labels.csv; images are direct children of images_dir."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rows = []
    for i in range(6):
        fname = f"img{i}.png"
        _make_tiny_png(images_dir / fname)
        rows.append({"filename": fname, "label": "cat" if i < 3 else "dog"})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, images_dir


@pytest.fixture
def class_structured_images_dir(tmp_path):
    """Class-subdir image tree matching ImageClassificationFromCSVDataset path logic.

    Images live at  images_dir / <label> / <filename>
    CSV rows:       filename,label
    """
    images_dir = tmp_path / "images"
    rows = []
    for cls in ("cat", "dog"):
        cls_dir = images_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(5):
            fname = f"{cls}_{i}.png"
            _make_tiny_png(cls_dir / fname)
            rows.append({"filename": fname, "label": cls})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, images_dir


@pytest.fixture
def fake_optuna_result():
    return {
        "best_value": 0.123,
        "best_params": {"lr": 0.001, "model_id": "google/efficientnet-b0"},
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
