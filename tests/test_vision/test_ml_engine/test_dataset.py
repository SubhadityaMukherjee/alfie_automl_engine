"""Tests for app/vision_automl/ml_engine/dataset.py."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from app.vision_automl.ml_engine.dataset import ImageClassificationFromCSVDataset


def _make_df():
    return pd.DataFrame(
        {
            "filename": ["a.png", "b.png", "c.png", "d.png"],
            "label": ["cat", "cat", "dog", "dog"],
        }
    )


# ---------------------------------------------------------------------------
# __init__ — no disk I/O
# ---------------------------------------------------------------------------


def test_init_from_dataframe():
    df = _make_df()
    ds = ImageClassificationFromCSVDataset(
        csv_file=df, root_dir=Path("/irrelevant"), img_col="filename", label_col="label"
    )
    assert len(ds) == 4
    assert ds.classes == ["cat", "dog"]
    assert ds.class_to_idx == {"cat": 0, "dog": 1}
    assert ds.idx_to_class == {0: "cat", 1: "dog"}


def test_init_from_csv_path(tmp_path):
    df = _make_df()
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    ds = ImageClassificationFromCSVDataset(
        csv_file=csv_path, root_dir=tmp_path, img_col="filename", label_col="label"
    )
    assert len(ds) == 4
    assert ds.classes == ["cat", "dog"]


def test_init_invalid_csv_file_type():
    with pytest.raises(ValueError, match="path or DataFrame"):
        ImageClassificationFromCSVDataset(csv_file=42, root_dir=Path("/x"))


def test_len():
    df = _make_df()
    ds = ImageClassificationFromCSVDataset(
        csv_file=df, root_dir=Path("/x"), img_col="filename", label_col="label"
    )
    assert len(ds) == len(df)


def test_labels_encoded_as_integers():
    df = _make_df()
    ds = ImageClassificationFromCSVDataset(
        csv_file=df, root_dir=Path("/x"), img_col="filename", label_col="label"
    )
    # After encoding, the label column should contain numeric values
    assert pd.api.types.is_numeric_dtype(ds.label_csv["label"])


def test_integer_labels_in_dataframe():
    """Integer labels still build a class_to_idx mapping (same if-branch is used)."""
    df = pd.DataFrame(
        {"img": ["a.png", "b.png", "c.png", "d.png"], "lbl": [0, 0, 1, 1]}
    )
    ds = ImageClassificationFromCSVDataset(
        csv_file=df, root_dir=Path("/x"), img_col="img", label_col="lbl"
    )
    assert len(ds) == 4
    assert 0 in ds.classes
    assert 1 in ds.classes


def test_classes_are_sorted():
    df = pd.DataFrame(
        {
            "filename": ["a.png", "b.png", "c.png"],
            "label": ["zebra", "ant", "monkey"],
        }
    )
    ds = ImageClassificationFromCSVDataset(
        csv_file=df, root_dir=Path("/x"), img_col="filename", label_col="label"
    )
    assert ds.classes == ["ant", "monkey", "zebra"]


def test_class_to_idx_and_idx_to_class_are_inverses():
    df = _make_df()
    ds = ImageClassificationFromCSVDataset(
        csv_file=df, root_dir=Path("/x"), img_col="filename", label_col="label"
    )
    for cls, idx in ds.class_to_idx.items():
        assert ds.idx_to_class[idx] == cls


# ---------------------------------------------------------------------------
# __getitem__ — requires real images on disk
# ---------------------------------------------------------------------------


@pytest.mark.full
def test_getitem_returns_image_and_tensor(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    ds = ImageClassificationFromCSVDataset(
        csv_file=csv_path, root_dir=images_dir, img_col="filename", label_col="label"
    )
    from PIL import Image

    img, label = ds[0]
    assert isinstance(img, Image.Image)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long


@pytest.mark.full
def test_getitem_label_in_valid_range(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    ds = ImageClassificationFromCSVDataset(
        csv_file=csv_path, root_dir=images_dir, img_col="filename", label_col="label"
    )
    for i in range(len(ds)):
        _, label = ds[i]
        assert 0 <= label.item() < len(ds.classes)


@pytest.mark.full
def test_getitem_file_not_found_raises(tmp_path):
    """Missing image raises FileNotFoundError with helpful message."""
    df = pd.DataFrame({"filename": ["missing.png"], "label": ["cat"]})
    (tmp_path / "cat").mkdir()  # class dir exists but image does not
    ds = ImageClassificationFromCSVDataset(
        csv_file=df, root_dir=tmp_path, img_col="filename", label_col="label"
    )
    with pytest.raises(FileNotFoundError, match="Image not found"):
        ds[0]


@pytest.mark.full
def test_getitem_with_transform(class_structured_images_dir):
    from torchvision import transforms as T

    csv_path, images_dir = class_structured_images_dir
    transform = T.Compose([T.ToTensor()])
    ds = ImageClassificationFromCSVDataset(
        csv_file=csv_path,
        root_dir=images_dir,
        img_col="filename",
        label_col="label",
        transform=transform,
    )
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
