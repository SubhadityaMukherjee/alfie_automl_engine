"""Fixtures for ml_engine tests."""

import pandas as pd
import pytest
from PIL import Image


def _make_tiny_png(path):
    Image.new("RGB", (10, 10), color=(64, 128, 32)).save(path)


@pytest.fixture
def large_class_structured_dir(tmp_path):
    """30 images per class (60 total) — enough for stratified splits."""
    images_dir = tmp_path / "images"
    rows = []
    for cls in ("cat", "dog"):
        cls_dir = images_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(30):
            fname = f"{cls}_{i}.png"
            _make_tiny_png(cls_dir / fname)
            rows.append({"filename": fname, "label": cls})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, images_dir
