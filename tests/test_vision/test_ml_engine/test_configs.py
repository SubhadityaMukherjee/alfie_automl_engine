"""Tests for app/vision_automl/ml_engine/configs/__init__.py"""

import pytest

from app.vision_automl.ml_engine.configs import (
    SUPPORTED_TASK_TYPES,
    load_task_config,
)

ALL_TASK_TYPES = [
    "image_classification",
    "image_segmentation",
    "object_detection",
    "video_classification",
    "keypoint_detection",
    "audio_classification",
    "text_classification",
    "question_answering",
    "causal_lm",
    "seq2seq_lm",
    "masked_lm",
]

REQUIRED_KEYS = {
    "small_models",
    "medium_models",
    "large_models",
    "lr_low",
    "lr_high",
    "batch_sizes",
    "weight_decay_low",
    "weight_decay_high",
    "max_epochs",
    "early_stopping_patience",
}


def test_supported_task_types_contains_all():
    assert set(ALL_TASK_TYPES) == SUPPORTED_TASK_TYPES


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_load_task_config_returns_dict(task_type):
    config = load_task_config(task_type)
    assert isinstance(config, dict)


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_load_task_config_has_required_keys(task_type):
    config = load_task_config(task_type)
    missing = REQUIRED_KEYS - set(config.keys())
    assert not missing, f"{task_type} config missing keys: {missing}"


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_load_task_config_model_lists_are_nonempty(task_type):
    config = load_task_config(task_type)
    for size in ("small", "medium", "large"):
        assert len(config[f"{size}_models"]) > 0, f"{task_type}: {size}_models is empty"


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_load_task_config_lr_bounds_are_ordered(task_type):
    config = load_task_config(task_type)
    assert config["lr_low"] < config["lr_high"]


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_load_task_config_batch_sizes_are_positive_ints(task_type):
    config = load_task_config(task_type)
    for bs in config["batch_sizes"]:
        assert isinstance(bs, int) and bs > 0


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_load_task_config_epochs_and_patience_are_positive(task_type):
    config = load_task_config(task_type)
    assert config["max_epochs"] > 0
    assert config["early_stopping_patience"] > 0


def test_load_task_config_raises_for_unknown_task():
    with pytest.raises(ValueError, match="Unknown task type"):
        load_task_config("flying_toaster")


def test_load_task_config_raises_with_helpful_message():
    with pytest.raises(ValueError, match="image_classification"):
        load_task_config("unknown")
