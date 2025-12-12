from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ImageTask(BaseModel):
    """Base Pydantic model describing common image task inputs."""

    train_dir: Path
    test_dir: Path | None = None
    label_format: Literal["folder", "csv"] = "folder"
    labels_file: Path | None = None  # used if label_format != 'folder'

    class Config:
        arbitrary_types_allowed = True


class ImageClassificationTask(ImageTask):
    """Configuration for single-label image classification tasks."""

    task_type: str = "image_classification"


# For the future
class ImageMultiLabelClassificationTask(ImageTask):
    """Configuration for multi-label image classification tasks."""

    task_type: str = "image_multilabel_classification"
    label_format: Literal["csv", "json"] = "csv"  # required


# For the future
class ImageRegressionTask(ImageTask):
    """Configuration for image regression tasks (predict numeric values)."""

    task_type: str = "image_regression"
    label_format: Literal["csv"] = "csv"  # regression needs exact values
