"""Shared fixtures for the whole test tree.

These live at the root of ``tests/`` so every test package can use them;
package-local ``conftest.py`` files override them where a variant is needed.
(CLI options live in the project-root ``conftest.py``.)
"""

import pytest
from PIL import Image


def _make_tiny_png(path) -> None:
    img = Image.new("RGB", (10, 10), color=(128, 64, 32))
    img.save(path)


@pytest.fixture
def synthetic_images_dir(tmp_path):
    """Flat images dir with labels.csv; images are direct children of images_dir."""
    import pandas as pd

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
    """Class-subdir image tree matching the torch datasets' path logic.

    Images live at  images_dir / <label> / <filename>
    CSV rows:       filename,label
    """
    import pandas as pd

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
