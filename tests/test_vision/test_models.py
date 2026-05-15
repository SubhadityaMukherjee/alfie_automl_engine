"""Tests for app/vision_automl/models.py Pydantic schemas."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.schemas.ml_tasks import (
    AudioClassificationTask,
    CausalLMTask,
    ImageClassificationTask,
    ImageMultiLabelClassificationTask,
    ImageRegressionTask,
    ImageSegmentationTask,
    ImageTask,
    KeypointDetectionTask,
    MaskedLMTask,
    ObjectDetectionTask,
    QuestionAnsweringTask,
    Seq2SeqLMTask,
    SequenceClassificationTask,
    SUPPORTED_VISION_TASK_TYPES,
    VideoClassificationTask,
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
    task = ImageTask(
        train_dir=Path("x"), label_format="csv", labels_file=Path("labels.csv")
    )
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


# ---------------------------------------------------------------------------
# New image task schemas
# ---------------------------------------------------------------------------


def test_image_segmentation_task_type():
    task = ImageSegmentationTask(train_dir=Path("x"))
    assert task.task_type == "image_segmentation"


def test_image_segmentation_inherits_image_task_defaults():
    task = ImageSegmentationTask(train_dir=Path("x"))
    assert task.label_format == "folder"
    assert task.test_dir is None


def test_object_detection_task_type():
    task = ObjectDetectionTask(train_dir=Path("x"))
    assert task.task_type == "object_detection"


def test_object_detection_requires_csv_format():
    task = ObjectDetectionTask(train_dir=Path("x"))
    assert task.label_format == "csv"


def test_video_classification_task_type():
    task = VideoClassificationTask(train_dir=Path("x"))
    assert task.task_type == "video_classification"


def test_video_classification_requires_csv_format():
    task = VideoClassificationTask(train_dir=Path("x"))
    assert task.label_format == "csv"


def test_keypoint_detection_task_type():
    task = KeypointDetectionTask(train_dir=Path("x"))
    assert task.task_type == "keypoint_detection"


# ---------------------------------------------------------------------------
# Audio task schema
# ---------------------------------------------------------------------------


def test_audio_classification_task_type():
    task = AudioClassificationTask(
        audio_dir=Path("audio/"), labels_file=Path("labels.csv")
    )
    assert task.task_type == "audio_classification"


def test_audio_classification_task_stores_paths():
    task = AudioClassificationTask(
        audio_dir=Path("audio/"), labels_file=Path("labels.csv")
    )
    assert task.audio_dir == Path("audio/")
    assert task.labels_file == Path("labels.csv")


def test_audio_classification_task_missing_audio_dir():
    with pytest.raises(ValidationError):
        AudioClassificationTask(labels_file=Path("labels.csv"))


# ---------------------------------------------------------------------------
# Text task schemas
# ---------------------------------------------------------------------------


def test_sequence_classification_task_type():
    task = SequenceClassificationTask(data_file=Path("data.csv"))
    assert task.task_type == "text_classification"


def test_question_answering_task_type():
    task = QuestionAnsweringTask(data_file=Path("data.csv"))
    assert task.task_type == "question_answering"


def test_causal_lm_task_type():
    task = CausalLMTask(data_file=Path("data.csv"))
    assert task.task_type == "causal_lm"


def test_seq2seq_lm_task_type():
    task = Seq2SeqLMTask(data_file=Path("data.csv"))
    assert task.task_type == "seq2seq_lm"


def test_masked_lm_task_type():
    task = MaskedLMTask(data_file=Path("data.csv"))
    assert task.task_type == "masked_lm"


def test_text_task_path_coercion():
    task = SequenceClassificationTask(data_file="data.csv")
    assert isinstance(task.data_file, Path)


# ---------------------------------------------------------------------------
# SUPPORTED_VISION_TASK_TYPES
# ---------------------------------------------------------------------------


def test_supported_vision_task_types_is_frozenset():
    assert isinstance(SUPPORTED_VISION_TASK_TYPES, frozenset)


def test_supported_vision_task_types_contains_all_slugs():
    expected = {
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
    }
    assert SUPPORTED_VISION_TASK_TYPES == expected
