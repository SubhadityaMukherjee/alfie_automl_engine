"""Tests for MultimodalClassificationDataModule and _infer_column_types."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from app.core.exceptions import AutoMLRuntimeError
from app.vision_automl.ml_engine.datamodule import (
    MultimodalClassificationDataModule,
    _infer_column_types,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_processor():
    processor = MagicMock()
    processor.return_value.pixel_values = torch.zeros(1, 3, 224, 224)
    processor.side_effect = lambda images, return_tensors: MagicMock(
        pixel_values=torch.zeros(len(images), 3, 224, 224)
    )
    return processor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multimodal_data_module(multimodal_class_structured_dir):
    csv_path, images_dir = multimodal_class_structured_dir
    with patch(
        "app.vision_automl.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = MultimodalClassificationDataModule(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            auxiliary_columns=["age", "weight", "color"],
            batch_size=4,
            num_workers=0,
        )
    return dm


# ---------------------------------------------------------------------------
# _infer_column_types
# ---------------------------------------------------------------------------


def test_infer_column_types_numeric_only():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4], "c": ["x", "y"]})
    numeric, categorical = _infer_column_types(df, ["a", "b"])
    assert numeric == ["a", "b"]
    assert categorical == []


def test_infer_column_types_categorical_only():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": ["p", "q"]})
    numeric, categorical = _infer_column_types(df, ["b", "c"])
    assert numeric == []
    assert categorical == ["b", "c"]


def test_infer_column_types_mixed():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["x", "y"], "other": [3, 4]})
    numeric, categorical = _infer_column_types(df, ["num", "cat", "other"])
    assert set(numeric) == {"num", "other"}
    assert categorical == ["cat"]


def test_infer_column_types_missing_columns_skipped():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    numeric, categorical = _infer_column_types(df, ["a", "nonexistent", "c"])
    assert numeric == ["a"]
    assert categorical == []


# ---------------------------------------------------------------------------
# Initialisation / setup
# ---------------------------------------------------------------------------


def test_setup_creates_all_three_datasets(multimodal_data_module):
    assert multimodal_data_module.train_dataset is not None
    assert multimodal_data_module.val_dataset is not None
    assert multimodal_data_module.test_dataset is not None


def test_num_classes_detected(multimodal_data_module):
    assert multimodal_data_module.num_classes == 2


def test_aux_feature_dim(multimodal_data_module):
    assert multimodal_data_module.aux_feature_dim == 3


def test_numeric_and_categorical_cols_split(multimodal_data_module):
    assert set(multimodal_data_module.numeric_cols) == {"age", "weight"}
    assert multimodal_data_module.categorical_cols == ["color"]


def test_id2label_and_label2id_populated(multimodal_data_module):
    assert len(multimodal_data_module.id2label) == 2
    assert len(multimodal_data_module.label2id) == 2


def test_label_mappings_are_inverses(multimodal_data_module):
    for idx, cls in multimodal_data_module.id2label.items():
        assert multimodal_data_module.label2id[cls] == idx


def test_processor_initialized(multimodal_data_module):
    assert multimodal_data_module.processor is not None


def test_split_sizes_sum_to_total(multimodal_class_structured_dir):
    csv_path, images_dir = multimodal_class_structured_dir
    with patch(
        "app.vision_automl.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = MultimodalClassificationDataModule(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            auxiliary_columns=["age", "weight", "color"],
        )

    total = len(pd.read_csv(csv_path))
    combined = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    assert combined == total


def test_no_auxiliary_columns_sets_dim_zero(multimodal_class_structured_dir):
    csv_path, images_dir = multimodal_class_structured_dir
    with patch(
        "app.vision_automl.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = MultimodalClassificationDataModule(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            auxiliary_columns=[],
        )
    assert dm.aux_feature_dim == 0
    assert dm.numeric_cols == []
    assert dm.categorical_cols == []


# ---------------------------------------------------------------------------
# _encode_auxiliary
# ---------------------------------------------------------------------------


def test_encode_auxiliary_fit_creates_scaler_and_encoder(multimodal_data_module):
    assert multimodal_data_module.scaler is not None
    assert multimodal_data_module.encoder is not None


def test_encode_auxiliary_numeric_scaling(multimodal_data_module):
    df = pd.DataFrame(
        {
            "age": [2.0, 4.0, 6.0, 8.0],
            "weight": [5.0, 10.0, 15.0, 20.0],
            "color": ["brown", "black", "brown", "black"],
        }
    )
    result = multimodal_data_module._encode_auxiliary(df, fit=False)
    mean_age = result["age"].mean()
    assert abs(mean_age) < 1.0


def test_encode_auxiliary_raises_if_scaler_not_fitted():
    dm = MultimodalClassificationDataModule.__new__(MultimodalClassificationDataModule)
    dm.numeric_cols = ["age"]
    dm.categorical_cols = []
    dm.scaler = None
    dm.encoder = None
    df = pd.DataFrame({"age": [1.0, 2.0]})
    with pytest.raises(AutoMLRuntimeError, match="Scaler not fitted"):
        dm._encode_auxiliary(df, fit=False)


def test_encode_auxiliary_raises_if_encoder_not_fitted():
    dm = MultimodalClassificationDataModule.__new__(MultimodalClassificationDataModule)
    dm.numeric_cols = []
    dm.categorical_cols = ["color"]
    dm.scaler = None
    dm.encoder = None
    df = pd.DataFrame({"color": ["red", "blue"]})
    with pytest.raises(AutoMLRuntimeError, match="OrdinalEncoder not fitted"):
        dm._encode_auxiliary(df, fit=False)


def test_encode_auxiliary_categorical_unknown_value_gets_minus_one():
    dm = MultimodalClassificationDataModule.__new__(MultimodalClassificationDataModule)
    dm.numeric_cols = []
    dm.categorical_cols = ["color"]
    dm.scaler = None
    train_df = pd.DataFrame({"color": ["red", "blue", "red"]})
    dm.encoder = MagicMock()
    dm.encoder.transform.return_value = np.array([[0.0], [1.0], [-1.0]])
    dm._encode_auxiliary(train_df, fit=True)
    result = dm._encode_auxiliary(
        pd.DataFrame({"color": ["red", "green", "yellow"]}), fit=False
    )
    assert result is not None


# ---------------------------------------------------------------------------
# _collate_fn
# ---------------------------------------------------------------------------


def test_collate_fn_returns_pixel_values_aux_and_labels(multimodal_data_module):
    images = [Image.new("RGB", (10, 10)) for _ in range(3)]
    aux_values = [np.array([1.0, 5.0, 0.0], dtype=np.float32) for _ in range(3)]
    labels = [0, 1, 0]
    batch = list(zip(images, aux_values, labels))
    result = multimodal_data_module._collate_fn(batch)

    assert "pixel_values" in result
    assert "aux_features" in result
    assert "labels" in result
    assert result["labels"].dtype == torch.long
    assert result["labels"].shape == (3,)
    assert result["aux_features"].dtype == torch.float32
    assert result["aux_features"].shape == (3, 3)


def test_collate_fn_empty_aux_features(multimodal_data_module):
    images = [Image.new("RGB", (10, 10)) for _ in range(2)]
    aux_values = [np.array([], dtype=np.float32) for _ in range(2)]
    labels = [0, 1]
    batch = list(zip(images, aux_values, labels))
    result = multimodal_data_module._collate_fn(batch)
    assert result["aux_features"].shape == (2, 0)


def test_collate_fn_raises_if_processor_none(multimodal_data_module):
    multimodal_data_module.processor = None
    batch = [(Image.new("RGB", (10, 10)), np.array([1.0], dtype=np.float32), 0)]
    with pytest.raises(AutoMLRuntimeError, match="Processor not initialized"):
        multimodal_data_module._collate_fn(batch)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def test_train_dataloader_returns_dataloader(multimodal_data_module):
    dl = multimodal_data_module.train_dataloader()
    assert isinstance(dl, DataLoader)


def test_val_dataloader_returns_dataloader(multimodal_data_module):
    dl = multimodal_data_module.val_dataloader()
    assert isinstance(dl, DataLoader)


def test_test_dataloader_returns_dataloader(multimodal_data_module):
    dl = multimodal_data_module.test_dataloader()
    assert isinstance(dl, DataLoader)


def test_val_dataloader_not_shuffled(multimodal_data_module):
    dl = multimodal_data_module.val_dataloader()
    assert dl.sampler.__class__.__name__ == "SequentialSampler"


def test_test_dataloader_not_shuffled(multimodal_data_module):
    dl = multimodal_data_module.test_dataloader()
    assert dl.sampler.__class__.__name__ == "SequentialSampler"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_train_dataloader_raises_if_dataset_none(multimodal_data_module):
    multimodal_data_module.train_dataset = None
    with pytest.raises(AutoMLRuntimeError, match="Train dataset not initialized"):
        multimodal_data_module.train_dataloader()


def test_val_dataloader_raises_if_dataset_none(multimodal_data_module):
    multimodal_data_module.val_dataset = None
    with pytest.raises(AutoMLRuntimeError, match="Validation dataset not initialized"):
        multimodal_data_module.val_dataloader()


def test_test_dataloader_raises_if_dataset_none(multimodal_data_module):
    multimodal_data_module.test_dataset = None
    with pytest.raises(AutoMLRuntimeError, match="Test dataset not initialized"):
        multimodal_data_module.test_dataloader()
