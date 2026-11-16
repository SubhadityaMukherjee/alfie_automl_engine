"""Tests for the default image/video augmentation pipelines and their
train-split-only wiring in the datamodules."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import v2

from app.ml_engine.augmentation import (
    build_image_augmentation,
    build_video_augmentation,
)
from app.ml_engine.datamodule import (
    ClassificationData,
    KeypointDetectionDataModule,
    MultimodalClassificationDataModule,
    VideoClassificationDataModule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NATIVE_HW = (36, 48)  # (H, W)
NUM_FRAMES = 4


def _gradient_image(size: int = 64) -> Image.Image:
    """A non-uniform image so any geometric change alters pixel values."""
    img = Image.new("RGB", (size, size))
    img.putdata(
        [
            (x * 3 % 256, y * 4 % 256, (x + y) * 2 % 256)
            for y in range(size)
            for x in range(size)
        ]
    )
    return img


def _make_mock_processor():
    processor = MagicMock()
    processor.return_value.pixel_values = torch.zeros(1, 3, 224, 224)
    processor.side_effect = lambda images, return_tensors: MagicMock(
        pixel_values=torch.zeros(len(images), 3, 224, 224)
    )
    return processor


def _fake_clip() -> torch.Tensor:
    """A deterministic pseudo-video frame stack, as ``_decode_clip_streaming``
    would return it: (T, C, H, W) uint8."""
    generator = torch.Generator().manual_seed(0)
    return (torch.rand(NUM_FRAMES, 3, *NATIVE_HW, generator=generator) * 255).to(
        torch.uint8
    )


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def test_image_augmentation_returns_rgb_pil_image():
    out = build_image_augmentation()(_gradient_image())
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"


def test_image_augmentation_is_stochastic():
    img = _gradient_image()
    pipeline = build_image_augmentation()
    torch.manual_seed(0)
    outputs = [tuple(pipeline(img).getdata()) for _ in range(16)]
    assert len(set(outputs)) > 1


def test_image_augmentation_uses_v2_stages():
    stages = build_image_augmentation().transforms
    assert any(isinstance(s, v2.RandomApply) for s in stages)  # pad
    assert any(isinstance(s, v2.RandomPerspective) for s in stages)
    assert any(isinstance(s, v2.RandomChoice) for s in stages)  # resize
    assert any(isinstance(s, v2.RandomGrayscale) for s in stages)


def test_video_augmentation_preserves_native_shape_and_range():
    torch.manual_seed(0)
    clip = torch.rand(NUM_FRAMES, 3, *NATIVE_HW)
    out = build_video_augmentation(NATIVE_HW)(clip)
    assert out.shape == clip.shape
    assert out.dtype == clip.dtype
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_video_augmentation_is_stochastic():
    torch.manual_seed(0)
    clip = torch.rand(NUM_FRAMES, 3, *NATIVE_HW)
    pipeline = build_video_augmentation(NATIVE_HW)
    outputs = [pipeline(clip) for _ in range(8)]
    assert any(not torch.equal(o, outputs[0]) for o in outputs[1:])


# ---------------------------------------------------------------------------
# Image datamodule wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def image_dm(large_class_structured_dir):
    csv_path, images_dir = large_class_structured_dir
    with patch(
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        yield ClassificationData(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            batch_size=4,
            num_workers=0,
        )


def test_image_datamodule_augments_train_split_only(image_dm):
    assert isinstance(image_dm.train_dataset.transform, v2.Compose)
    assert image_dm.val_dataset.transform is None
    assert image_dm.test_dataset.transform is None


def test_image_datamodule_train_samples_survive_augmentation(image_dm):
    sample, label = image_dm.train_dataset[0]
    assert isinstance(sample, Image.Image)
    assert sample.mode == "RGB"
    assert label.dtype == torch.long


def test_image_datamodule_augment_disabled_leaves_transforms_none(
    large_class_structured_dir,
):
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
            num_workers=0,
            augment=False,
        )
    assert dm.train_dataset.transform is None


def test_image_datamodule_explicit_transform_wins(large_class_structured_dir):
    csv_path, images_dir = large_class_structured_dir
    sentinel = T.CenterCrop(4)
    with patch(
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = ClassificationData(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            num_workers=0,
            transform=sentinel,
        )
    assert dm.train_dataset.transform is sentinel
    assert dm.val_dataset.transform is sentinel


def test_keypoint_datamodule_disables_default_augmentation(
    large_class_structured_dir,
):
    """Keypoints are absolute pixels: warping images would corrupt labels."""
    csv_path, images_dir = large_class_structured_dir
    with patch(
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = KeypointDetectionDataModule(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            num_workers=0,
        )
    assert dm._augment_enabled() is False
    assert dm.train_dataset.transform is None


def test_multimodal_datamodule_augments_train_split_only(
    multimodal_class_structured_dir,
):
    csv_path, images_dir = multimodal_class_structured_dir
    with patch(
        "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
        return_value=_make_mock_processor(),
    ):
        dm = MultimodalClassificationDataModule(
            csv_file=csv_path,
            root_dir=images_dir,
            img_col="filename",
            label_col="label",
            auxiliary_columns=["age", "weight", "color"],
            num_workers=0,
        )
    assert isinstance(dm.train_dataset.transform, v2.Compose)
    assert dm.val_dataset.transform is None
    assert dm.test_dataset.transform is None


# ---------------------------------------------------------------------------
# Video datamodule wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def video_dm(tmp_path):
    rows = [
        {"video_path": f"clip_{i}.mp4", "label": i % 2, "class_labels": f"[{i % 2}]"}
        for i in range(20)
    ]
    csv_path = tmp_path / "videos.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with (
        patch(
            "app.ml_engine.datamodule.AutoImageProcessor.from_pretrained",
            return_value=_make_mock_processor(),
        ),
        patch(
            "app.ml_engine.datamodule._decode_clip_streaming",
            return_value=_fake_clip(),
        ),
    ):
        yield VideoClassificationDataModule(
            csv_file=csv_path,
            root_dir=tmp_path,
            num_frames=NUM_FRAMES,
            batch_size=2,
            num_workers=0,
            val_split=0.2,
            test_split=0.1,
        )


def test_video_datamodule_augments_train_split_only(video_dm):
    train_dataset = video_dm._make_dataset(video_dm.train_df, augment=True)
    val_dataset = video_dm._make_dataset(video_dm.val_df, augment=False)

    torch.manual_seed(0)
    augmented = [train_dataset[0][0] for _ in range(8)]
    # Native size restored and values stay in the [0, 1] float range.
    for clip in augmented:
        assert clip.shape == (NUM_FRAMES, 3, *NATIVE_HW)
        assert clip.dtype == torch.float32
        assert float(clip.min()) >= 0.0 and float(clip.max()) <= 1.0
    # Stochastic: repeated draws of the same clip are not all identical.
    assert any(not torch.equal(c, augmented[0]) for c in augmented[1:])

    raw = _fake_clip().float() / 255.0
    assert torch.equal(val_dataset[0][0], raw)


def test_video_dataloaders_stack_augmented_clips(video_dm):
    """A full batch from the train loader must stack despite augmentation."""
    batch = next(iter(video_dm.train_dataloader()))
    assert batch["pixel_values"].shape == (2, NUM_FRAMES, 3, *NATIVE_HW)
    assert batch["labels"].shape == (2,)
