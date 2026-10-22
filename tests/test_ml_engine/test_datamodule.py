"""Tests for app/ml_engine/datamodule.py."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from app.core.exceptions import AutoMLDataError, AutoMLRuntimeError
from app.ml_engine.datamodule import (
    ClassificationData,
    ImageSegmentationDataModule,
    ImageSegmentationFromCSVDataset,
    KeypointDetectionDataModule,
    _evenly_spaced_timestamps,
    _processor_draft_size,
    auto_num_workers,
)

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
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
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
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
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
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
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
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
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
    with pytest.raises(AutoMLRuntimeError, match="Processor not initialized"):
        data_module._collate_fn(list(zip(images, labels)))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_train_dataloader_raises_if_dataset_none(data_module):
    data_module.train_dataset = None
    with pytest.raises(AutoMLRuntimeError, match="Train dataset not initialized"):
        data_module.train_dataloader()


def test_val_dataloader_raises_if_dataset_none(data_module):
    data_module.val_dataset = None
    with pytest.raises(AutoMLRuntimeError, match="Validation dataset not initialized"):
        data_module.val_dataloader()


def test_test_dataloader_raises_if_dataset_none(data_module):
    data_module.test_dataset = None
    with pytest.raises(AutoMLRuntimeError, match="Test dataset not initialized"):
        data_module.test_dataloader()


# ---------------------------------------------------------------------------
# Loader performance kwargs
# ---------------------------------------------------------------------------


def test_loader_kwargs_zero_workers_no_persistence(data_module):
    data_module.num_workers = 0
    kwargs = data_module._loader_kwargs()
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_loader_kwargs_positive_workers_persist_and_prefetch(data_module):
    data_module.num_workers = 2
    kwargs = data_module._loader_kwargs()
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4


def test_loader_kwargs_pin_memory_only_on_cuda(data_module, monkeypatch):
    data_module.num_workers = 0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "pin_memory" not in data_module._loader_kwargs()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert data_module._loader_kwargs()["pin_memory"] is True


# ---------------------------------------------------------------------------
# auto_num_workers
# ---------------------------------------------------------------------------


def test_auto_num_workers_uses_all_but_one_cpu(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert auto_num_workers() == 2


def test_auto_num_workers_respects_cap(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 100)
    assert auto_num_workers() == 8  # default auto_num_workers_max


def test_auto_num_workers_never_negative(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert auto_num_workers() == 0


def test_auto_num_workers_explicit_max(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert auto_num_workers(max_workers=3) == 3


# ---------------------------------------------------------------------------
# _processor_draft_size
# ---------------------------------------------------------------------------


def test_draft_size_from_int_size():
    assert _processor_draft_size(MagicMock(size=224)) == (448, 448)


def test_draft_size_from_shortest_edge_dict():
    assert _processor_draft_size(MagicMock(size={"shortest_edge": 224})) == (448, 448)


def test_draft_size_from_height_width_dict():
    assert _processor_draft_size(MagicMock(size={"height": 224, "width": 384})) == (
        448,
        448,
    )


def test_draft_size_none_when_size_unknown():
    assert _processor_draft_size(MagicMock(spec=[])) is None


def test_draft_size_disabled_via_settings(monkeypatch):
    monkeypatch.setenv("FAST_IMAGE_DECODE", "false")
    assert _processor_draft_size(MagicMock(size=224)) is None


# ---------------------------------------------------------------------------
# Draft decode gating per task
# ---------------------------------------------------------------------------


def test_classification_draft_decode_enabled():
    dm = ClassificationData.__new__(ClassificationData)
    assert dm._draft_decode_enabled() is True


def test_segmentation_masks_stay_pixel_aligned(tmp_path):
    """Segmentation has no draft-decode path: masks are read as-is, so they
    stay pixel-aligned with the source image (the guarantee the old
    ``_draft_decode_enabled`` flag provided for the classification layout).
    """
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(images_dir / "img_00.png")
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[4:, 6:] = 1
    Image.fromarray(mask).save(masks_dir / "mask_00.png")

    df = pd.DataFrame([{"filename": "img_00.png", "mask_filename": "mask_00.png"}])
    dataset = ImageSegmentationFromCSVDataset(
        csv_file=df, root_dir=tmp_path, img_col="filename", mask_col="mask_filename"
    )

    image, mask_tensor = dataset[0]
    assert image.size == (12, 8)
    assert mask_tensor.shape == (8, 12)
    assert mask_tensor.dtype == torch.long
    assert sorted(torch.unique(mask_tensor).tolist()) == [0, 1]


def test_segmentation_dataset_rejects_non_2d_mask(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "img_00.png")
    Image.new("RGB", (8, 8)).save(masks_dir / "mask_00.png")  # 3-channel mask

    df = pd.DataFrame([{"filename": "img_00.png", "mask_filename": "mask_00.png"}])
    dataset = ImageSegmentationFromCSVDataset(
        csv_file=df, root_dir=tmp_path, img_col="filename", mask_col="mask_filename"
    )

    with pytest.raises(AutoMLDataError, match="single channel"):
        dataset[0]


def test_segmentation_dataset_rejects_size_mismatch(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "img_00.png")
    Image.new("L", (4, 4)).save(masks_dir / "mask_00.png")

    df = pd.DataFrame([{"filename": "img_00.png", "mask_filename": "mask_00.png"}])
    dataset = ImageSegmentationFromCSVDataset(
        csv_file=df, root_dir=tmp_path, img_col="filename", mask_col="mask_filename"
    )

    with pytest.raises(AutoMLDataError, match="dimensions do not match"):
        dataset[0]


def test_segmentation_setup_derives_class_ids_from_masks(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    rows = []
    for i in range(6):
        Image.new("RGB", (8, 8)).save(images_dir / f"img_{i:02d}.png")
        mask = np.full((8, 8), i % 2, dtype=np.uint8)  # alternating IDs 0 / 1
        Image.fromarray(mask).save(masks_dir / f"mask_{i:02d}.png")
        rows.append(
            {"filename": f"img_{i:02d}.png", "mask_filename": f"mask_{i:02d}.png"}
        )
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with patch(
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = ImageSegmentationDataModule(
            csv_file=csv_path,
            root_dir=tmp_path,
            img_col="filename",
            mask_col="mask_filename",
            batch_size=2,
            num_workers=0,
        )

    assert dm.num_classes == 2
    assert dm.id2label == {0: "0", 1: "1"}
    assert dm.label2id == {"0": 0, "1": 1}
    batch = next(iter(dm.train_dataloader()))
    assert batch["pixel_values"].shape[0] == batch["labels"].shape[0]


def test_segmentation_setup_rejects_non_contiguous_class_ids(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    rows = []
    for i in range(4):
        Image.new("RGB", (8, 8)).save(images_dir / f"img_{i:02d}.png")
        mask = np.full((8, 8), 2, dtype=np.uint8)  # only ID 2 → not contiguous
        Image.fromarray(mask).save(masks_dir / f"mask_{i:02d}.png")
        rows.append(
            {"filename": f"img_{i:02d}.png", "mask_filename": f"mask_{i:02d}.png"}
        )
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with patch(
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        with pytest.raises(AutoMLDataError, match="contiguous"):
            ImageSegmentationDataModule(
                csv_file=csv_path,
                root_dir=tmp_path,
                img_col="filename",
                mask_col="mask_filename",
                batch_size=2,
                num_workers=0,
            )


def test_keypoint_draft_decode_disabled():
    dm = KeypointDetectionDataModule.__new__(KeypointDetectionDataModule)
    assert dm._draft_decode_enabled() is False


def test_setup_passes_decode_size_to_datasets(large_class_structured_dir):
    csv_path, images_dir = large_class_structured_dir
    processor = _make_mock_processor()
    processor.size = {"shortest_edge": 224}
    with patch(
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=processor,
    ):
        dm = ClassificationData(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
        )
    assert dm.train_dataset is not None
    assert dm.train_dataset.decode_size == (448, 448)


# ---------------------------------------------------------------------------
# Video timestamp sampling
# ---------------------------------------------------------------------------


def test_evenly_spaced_timestamps_basic():
    assert _evenly_spaced_timestamps(8.0, 4) == [0.0, 2.0, 4.0, 6.0]


def test_evenly_spaced_timestamps_single_frame():
    assert _evenly_spaced_timestamps(8.0, 1) == [0.0]


def test_evenly_spaced_timestamps_empty():
    assert _evenly_spaced_timestamps(8.0, 0) == []
