from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel


class ImageTask(BaseModel):
    """
    Base class for image tasks
    """

    train_dir: Path
    test_dir: Optional[Path] = None
    label_format: Literal["folder", "csv"] = "folder"
    labels_file: Optional[Path] = None  # used if label_format != 'folder'

    class Config:
        arbitrary_types_allowed = True


class ImageClassificationTask(ImageTask):
    """
    For single-label classification (e.g., cat vs dog vs X)
    """

    task_type: str = "image_classification"


# For the future
class ImageMultiLabelClassificationTask(ImageTask):
    """
    For multi-label classification (e.g., person AND dog in the same image)
    """

    task_type: str = "image_multilabel_classification"
    label_format: Literal["csv", "json"] = "csv"  # required


# For the future
class ImageRegressionTask(ImageTask):
    """
    Predict a numeric value from an image (e.g., predict age from face image)
    """

    task_type: str = "image_regression"
    label_format: Literal["csv"] = "csv"  # regression needs exact values
