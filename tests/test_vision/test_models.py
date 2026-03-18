"""Tests for app/vision_automl/models.py Pydantic schemas."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.vision_automl.models import (
    ImageClassificationTask,
    ImageMultiLabelClassificationTask,
    ImageRegressionTask,
    ImageTask,
)


def test_image_task_defaults():
    task = ImageTask(train_dir=Path("data/train"))
    assert task.train_dir == Path("data/train")
    assert task.test_dir is None
    assert task.label_format == "folder"
    assert task.labels_file is None


def test_image_task_with_test_dir():
    task = ImageTask(train_dir=Path("x"), test_dir=Path("y"))
    assert task.test_dir == Path("y")


def test_image_task_with_labels_file():
    task = ImageTask(train_dir=Path("x"), label_format="csv", labels_file=Path("labels.csv"))
    assert task.label_format == "csv"
    assert task.labels_file == Path("labels.csv")


def test_image_task_invalid_label_format():
    with pytest.raises(ValidationError):
        ImageTask(train_dir=Path("x"), label_format="xml")


def test_image_task_path_coercion():
    task = ImageTask(train_dir="data/train")
    assert isinstance(task.train_dir, Path)


def test_image_classification_task_type():
    task = ImageClassificationTask(train_dir=Path("x"))
    assert task.task_type == "image_classification"


def test_image_classification_task_inherits_defaults():
    task = ImageClassificationTask(train_dir=Path("x"))
    assert task.label_format == "folder"
    assert task.test_dir is None


def test_image_multilabel_task_type():
    task = ImageMultiLabelClassificationTask(train_dir=Path("x"))
    assert task.task_type == "image_multilabel_classification"
    assert task.label_format == "csv"


def test_image_multilabel_task_json_format():
    task = ImageMultiLabelClassificationTask(train_dir=Path("x"), label_format="json")
    assert task.label_format == "json"


def test_image_multilabel_task_rejects_folder_format():
    with pytest.raises(ValidationError):
        ImageMultiLabelClassificationTask(train_dir=Path("x"), label_format="folder")


def test_image_regression_task_type():
    task = ImageRegressionTask(train_dir=Path("x"))
    assert task.task_type == "image_regression"
    assert task.label_format == "csv"


def test_image_regression_task_rejects_folder_format():
    with pytest.raises(ValidationError):
        ImageRegressionTask(train_dir=Path("x"), label_format="folder")


def test_image_regression_task_rejects_json_format():
    with pytest.raises(ValidationError):
        ImageRegressionTask(train_dir=Path("x"), label_format="json")


@pytest.mark.parametrize(
    "task_cls, expected_type",
    [
        (ImageClassificationTask, "image_classification"),
        (ImageMultiLabelClassificationTask, "image_multilabel_classification"),
        (ImageRegressionTask, "image_regression"),
    ],
)
def test_task_type_is_string(task_cls, expected_type):
    task = task_cls(train_dir=Path("x"))
    assert isinstance(task.task_type, str)
    assert task.task_type == expected_type
