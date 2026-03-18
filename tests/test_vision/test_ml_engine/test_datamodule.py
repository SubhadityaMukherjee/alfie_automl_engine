"""Tests for app/vision_automl/ml_engine/datamodule.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from app.vision_automl.ml_engine.datamodule import ClassificationData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_processor():
    """Return a mock AutoImageProcessor that produces pixel_values tensors."""
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
def data_module(large_class_structured_dir):
    """ClassificationData with mocked HF processor."""
    csv_path, images_dir = large_class_structured_dir
    with patch(
        "app.vision_automl.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = ClassificationData(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            batch_size=4,
            num_workers=0,
        )
    return dm


# ---------------------------------------------------------------------------
# Initialisation / setup
# ---------------------------------------------------------------------------


def test_setup_creates_all_three_datasets(data_module):
    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert data_module.test_dataset is not None


def test_num_classes_detected(data_module):
    assert data_module.num_classes == 2


def test_id2label_and_label2id_populated(data_module):
    assert len(data_module.id2label) == 2
    assert len(data_module.label2id) == 2


def test_label_mappings_are_inverses(data_module):
    for idx, cls in data_module.id2label.items():
        assert data_module.label2id[cls] == idx


def test_processor_initialized(data_module):
    assert data_module.processor is not None


def test_split_sizes_sum_to_total(large_class_structured_dir):
    """train + val + test == total rows in CSV."""
    csv_path, images_dir = large_class_structured_dir
    with patch(
        "app.vision_automl.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = ClassificationData(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
        )

    total = len(pd.read_csv(csv_path))
    combined = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    assert combined == total


def test_val_split_respected(large_class_structured_dir):
    """Val set is roughly val_split fraction of the full dataset."""
    csv_path, images_dir = large_class_structured_dir
    val_split = 0.2
    with patch(
        "app.vision_automl.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = ClassificationData(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            val_split=val_split,
            test_split=0.1,
        )

    total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    assert abs(len(dm.val_dataset) / total - val_split) < 0.05


def test_processor_called_with_model_id(large_class_structured_dir):
    csv_path, images_dir = large_class_structured_dir
    model_id = "google/vit-base-patch16-224"
    with patch(
        "app.vision_automl.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
    ) as mock_from_pretrained:
        mock_from_pretrained.return_value = _make_mock_processor()
        ClassificationData(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            hf_model_id=model_id,
        )
    mock_from_pretrained.assert_called_once_with(model_id)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def test_train_dataloader_returns_dataloader(data_module):
    dl = data_module.train_dataloader()
    assert isinstance(dl, DataLoader)


def test_val_dataloader_returns_dataloader(data_module):
    dl = data_module.val_dataloader()
    assert isinstance(dl, DataLoader)


def test_test_dataloader_returns_dataloader(data_module):
    dl = data_module.test_dataloader()
    assert isinstance(dl, DataLoader)


def test_train_dataloader_batch_size(data_module):
    dl = data_module.train_dataloader()
    assert dl.batch_size == data_module.batch_size


def test_val_dataloader_not_shuffled(data_module):
    dl = data_module.val_dataloader()
    assert dl.sampler.__class__.__name__ == "SequentialSampler"


def test_test_dataloader_not_shuffled(data_module):
    dl = data_module.test_dataloader()
    assert dl.sampler.__class__.__name__ == "SequentialSampler"


# ---------------------------------------------------------------------------
# _collate_fn
# ---------------------------------------------------------------------------


def test_collate_fn_returns_pixel_values_and_labels(data_module):
    images = [Image.new("RGB", (10, 10)) for _ in range(3)]
    labels = [0, 1, 0]
    batch = list(zip(images, labels))
    result = data_module._collate_fn(batch)

    assert "pixel_values" in result
    assert "labels" in result
    assert result["labels"].dtype == torch.long
    assert result["labels"].shape == (3,)


def test_collate_fn_raises_if_processor_none(data_module):
    data_module.processor = None
    images = [Image.new("RGB", (10, 10))]
    labels = [0]
    with pytest.raises(RuntimeError, match="Processor not initialized"):
        data_module._collate_fn(list(zip(images, labels)))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_train_dataloader_raises_if_dataset_none(data_module):
    data_module.train_dataset = None
    with pytest.raises(RuntimeError, match="Train dataset not initialized"):
        data_module.train_dataloader()


def test_val_dataloader_raises_if_dataset_none(data_module):
    data_module.val_dataset = None
    with pytest.raises(RuntimeError, match="Validation dataset not initialized"):
        data_module.val_dataloader()


def test_test_dataloader_raises_if_dataset_none(data_module):
    data_module.test_dataset = None
    with pytest.raises(RuntimeError, match="Test dataset not initialized"):
        data_module.test_dataloader()
