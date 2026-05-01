"""Tests for MultimodalClassificationDataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.vision_automl.ml_engine.dataset import MultimodalClassificationDataset


def _make_df(with_aux=True):
    data = {
        "filename": ["a.png", "b.png", "c.png", "d.png"],
        "label": ["cat", "cat", "dog", "dog"],
    }
    if with_aux:
        data["age"] = [2.0, 3.5, 5.0, 7.0]
        data["color"] = ["brown", "black", "brown", "black"]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# __init__ — no disk I/O
# ---------------------------------------------------------------------------


def test_init_from_dataframe():
    df = _make_df()
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=Path("/irrelevant"),
        img_col="filename",
        label_col="label",
        auxiliary_columns=["age", "color"],
    )
    assert len(ds) == 4
    assert ds.classes == ["cat", "dog"]
    assert ds.class_to_idx == {"cat": 0, "dog": 1}
    assert ds.auxiliary_columns == ["age", "color"]


def test_init_from_csv_path(tmp_path):
    df = _make_df()
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    ds = MultimodalClassificationDataset(
        csv_file=csv_path,
        root_dir=tmp_path,
        img_col="filename",
        label_col="label",
        auxiliary_columns=["age", "color"],
    )
    assert len(ds) == 4


def test_init_without_auxiliary_columns():
    df = _make_df(with_aux=False)
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=Path("/x"),
        img_col="filename",
        label_col="label",
    )
    assert ds.auxiliary_columns == []


def test_init_invalid_csv_file_type():
    with pytest.raises(ValueError, match="path or DataFrame"):
        MultimodalClassificationDataset(csv_file=42, root_dir=Path("/x"))


def test_classes_are_sorted():
    df = pd.DataFrame(
        {
            "filename": ["a.png", "b.png", "c.png"],
            "label": ["zebra", "ant", "monkey"],
            "weight": [10.0, 5.0, 8.0],
        }
    )
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=Path("/x"),
        img_col="filename",
        label_col="label",
        auxiliary_columns=["weight"],
    )
    assert ds.classes == ["ant", "monkey", "zebra"]


def test_class_to_idx_and_idx_to_class_are_inverses():
    df = _make_df()
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=Path("/x"),
        img_col="filename",
        label_col="label",
        auxiliary_columns=["age", "color"],
    )
    for cls, idx in ds.class_to_idx.items():
        assert ds.idx_to_class[idx] == cls


def test_len():
    df = _make_df()
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=Path("/x"),
        img_col="filename",
        label_col="label",
        auxiliary_columns=["age", "color"],
    )
    assert len(ds) == len(df)


# ---------------------------------------------------------------------------
# __getitem__ — requires real images on disk
# ---------------------------------------------------------------------------


@pytest.mark.full
def test_getitem_returns_image_aux_and_label(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["age"] = [2.0, 3.0, 5.0, 7.0, 1.0, 4.0, 6.0, 8.0, 2.5, 3.5]
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=images_dir,
        img_col="filename",
        label_col="label",
        auxiliary_columns=["age"],
    )
    from PIL import Image

    img, aux, label = ds[0]
    assert isinstance(img, Image.Image)
    assert isinstance(aux, np.ndarray)
    assert aux.dtype == np.float32
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long


@pytest.mark.full
def test_getitem_label_in_valid_range(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["weight"] = range(len(df))
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=images_dir,
        img_col="filename",
        label_col="label",
        auxiliary_columns=["weight"],
    )
    for i in range(len(ds)):
        _, _, label = ds[i]
        assert 0 <= label.item() < len(ds.classes)


@pytest.mark.full
def test_getitem_aux_values_shape(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["age"] = [1.0] * len(df)
    df["weight"] = [5.0] * len(df)
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=images_dir,
        img_col="filename",
        label_col="label",
        auxiliary_columns=["age", "weight"],
    )
    _, aux, _ = ds[0]
    assert aux.shape == (2,)


@pytest.mark.full
def test_getitem_empty_aux_columns_returns_empty_array(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    ds = MultimodalClassificationDataset(
        csv_file=csv_path,
        root_dir=images_dir,
        img_col="filename",
        label_col="label",
        auxiliary_columns=[],
    )
    _, aux, _ = ds[0]
    assert aux.shape == (0,)
    assert aux.dtype == np.float32


@pytest.mark.full
def test_getitem_no_auxiliary_columns_returns_empty_array(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    ds = MultimodalClassificationDataset(
        csv_file=csv_path,
        root_dir=images_dir,
        img_col="filename",
        label_col="label",
    )
    _, aux, _ = ds[0]
    assert aux.shape == (0,)


@pytest.mark.full
def test_getitem_file_not_found_raises(tmp_path):
    df = pd.DataFrame({"filename": ["missing.png"], "label": ["cat"], "age": [2.0]})
    (tmp_path / "cat").mkdir()
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=tmp_path,
        img_col="filename",
        label_col="label",
        auxiliary_columns=["age"],
    )
    with pytest.raises(FileNotFoundError, match="Image not found"):
        ds[0]


@pytest.mark.full
def test_getitem_with_transform(class_structured_images_dir):
    from torchvision import transforms as T

    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["age"] = range(len(df))
    transform = T.Compose([T.ToTensor()])
    ds = MultimodalClassificationDataset(
        csv_file=df,
        root_dir=images_dir,
        img_col="filename",
        label_col="label",
        auxiliary_columns=["age"],
        transform=transform,
    )
    img, _, _ = ds[0]
    assert isinstance(img, torch.Tensor)
