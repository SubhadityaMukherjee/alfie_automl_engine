import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from app.core.exceptions import AutoMLDataError, AutoMLValidationError

logger = logging.getLogger(__name__)


class ImageClassificationFromCSVDataset(Dataset):
    """Torch dataset that reads image paths and labels from a CSV/DataFrame."""

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        root_dir: Path,
        img_col: str = "image",
        label_col: str = "label",
        transform: Optional[T.Compose] = None,
    ):
        if isinstance(csv_file, Path):
            try:
                self.label_csv = pd.read_csv(csv_file)
            except FileNotFoundError:
                logger.error("Dataset CSV file not found: %s", csv_file)
                raise
            except pd.errors.EmptyDataError:
                logger.error("Dataset CSV file is empty: %s", csv_file)
                raise AutoMLDataError(f"Dataset CSV file is empty: {csv_file}")
            except pd.errors.ParserError as e:
                logger.error("Failed to parse dataset CSV file: %s", e)
                raise
            except Exception as e:
                logger.error("Unexpected error reading dataset CSV file: %s", e)
                raise
        elif isinstance(csv_file, pd.DataFrame):
            self.label_csv = csv_file.reset_index(drop=True)
        else:
            raise AutoMLValidationError("csv_file must be a path or DataFrame")

        self.root_dir = root_dir
        self.img_col = img_col
        self.label_col = label_col
        # By default, do not apply torchvision transforms so that a Hugging Face
        # AutoImageProcessor can handle preprocessing in a DataLoader collate_fn.
        self.transform = transform

        if self.label_csv[self.label_col].dtype not in [int, float]:
            self.classes = sorted(self.label_csv[self.label_col].unique().tolist())
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(self.classes)
            }
            self.idx_to_class = {
                idx: cls_name for cls_name, idx in self.class_to_idx.items()
            }
            self.label_csv[self.label_col] = self.label_csv[self.label_col].map(
                self.class_to_idx
            )
        else:
            self.classes = sorted(self.label_csv[self.label_col].unique().tolist())
            self.class_to_idx = {cls: cls for cls in self.classes}
            self.idx_to_class = {cls: cls for cls in self.classes}

    def __len__(self):
        """Return number of samples."""
        return len(self.label_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.label_csv.iloc[idx]
        label_idx = int(row[self.label_col])
        label_name = self.idx_to_class[label_idx]

        filename = str(row[self.img_col]).strip()

        img_path = self.root_dir / label_name / filename
        if not img_path.exists():
            logger.error(
                "Image not found: root_dir=%s, label_name=%s, filename=%s",
                self.root_dir,
                label_name,
                filename,
            )
            print(os.listdir(self.root_dir))
            print(os.listdir(self.root_dir / label_name))

            raise AutoMLDataError(
                f"Image not found\n"
                f"Expected path: {img_path}\n"
                f"root_dir: {self.root_dir}\n"
                f"label_name: {label_name}\n"
                f"filename: {repr(filename)}"
            )

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to open or convert image %s: %s", img_path, e)
            raise

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label_idx, dtype=torch.long)


class TextClassificationFromCSVDataset(Dataset):
    """Torch dataset that reads text and labels from a CSV/DataFrame.

    Expected columns: ``text`` (str) and ``label`` (str or int).
    Returns ``(text, label_idx)`` tuples — the collate function in the
    datamodule applies the tokeniser.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        text_col: str = "text",
        label_col: str = "label",
    ):
        if isinstance(csv_file, Path):
            try:
                self.df = pd.read_csv(csv_file)
            except FileNotFoundError:
                logger.error("Dataset CSV file not found: %s", csv_file)
                raise
            except pd.errors.EmptyDataError:
                logger.error("Dataset CSV file is empty: %s", csv_file)
                raise AutoMLDataError(f"Dataset CSV file is empty: {csv_file}")
            except pd.errors.ParserError as e:
                logger.error("Failed to parse dataset CSV file: %s", e)
                raise
            except Exception as e:
                logger.error("Unexpected error reading dataset CSV file: %s", e)
                raise
        elif isinstance(csv_file, pd.DataFrame):
            self.df = csv_file.reset_index(drop=True)
        else:
            raise AutoMLValidationError("csv_file must be a path or DataFrame")

        self.text_col = text_col
        self.label_col = label_col

        if self.df[self.label_col].dtype == object:
            self.classes = sorted(self.df[self.label_col].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.df = self.df.copy()
            self.df[self.label_col] = self.df[self.label_col].map(self.class_to_idx)
        else:
            self.classes = sorted(self.df[self.label_col].unique().tolist())
            self.class_to_idx = {c: c for c in self.classes}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        if torch.is_tensor(idx):
            idx = idx.item()
        row = self.df.iloc[idx]
        return str(row[self.text_col]), int(row[self.label_col])


class QuestionAnsweringFromCSVDataset(Dataset):
    """Dataset for extractive QA tasks.

    Expected CSV columns: ``question``, ``context``, ``answer_start`` (int),
    ``answer_text`` (str).  Returns raw strings; the datamodule tokenises them.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        question_col: str = "question",
        context_col: str = "context",
        answer_start_col: str = "answer_start",
        answer_text_col: str = "answer_text",
    ):
        if isinstance(csv_file, Path):
            try:
                self.df = pd.read_csv(csv_file)
            except FileNotFoundError:
                logger.error("Dataset CSV file not found: %s", csv_file)
                raise
            except pd.errors.EmptyDataError:
                logger.error("Dataset CSV file is empty: %s", csv_file)
                raise AutoMLDataError(f"Dataset CSV file is empty: {csv_file}")
            except pd.errors.ParserError as e:
                logger.error("Failed to parse dataset CSV file: %s", e)
                raise
            except Exception as e:
                logger.error("Unexpected error reading dataset CSV file: %s", e)
                raise
        elif isinstance(csv_file, pd.DataFrame):
            self.df = csv_file.reset_index(drop=True)
        else:
            raise AutoMLValidationError("csv_file must be a path or DataFrame")

        self.question_col = question_col
        self.context_col = context_col
        self.answer_start_col = answer_start_col
        self.answer_text_col = answer_text_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.item()
        row = self.df.iloc[idx]
        return {
            "question": str(row[self.question_col]),
            "context": str(row[self.context_col]),
            "answer_start": int(row[self.answer_start_col]),
            "answer_text": str(row[self.answer_text_col]),
        }


class Seq2SeqFromCSVDataset(Dataset):
    """Dataset for sequence-to-sequence tasks.

    Expected CSV columns: ``input_text`` and ``target_text``.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        input_col: str = "input_text",
        target_col: str = "target_text",
    ):
        if isinstance(csv_file, Path):
            try:
                self.df = pd.read_csv(csv_file)
            except FileNotFoundError:
                logger.error("Dataset CSV file not found: %s", csv_file)
                raise
            except pd.errors.EmptyDataError:
                logger.error("Dataset CSV file is empty: %s", csv_file)
                raise AutoMLDataError(f"Dataset CSV file is empty: {csv_file}")
            except pd.errors.ParserError as e:
                logger.error("Failed to parse dataset CSV file: %s", e)
                raise
            except Exception as e:
                logger.error("Unexpected error reading dataset CSV file: %s", e)
                raise
        elif isinstance(csv_file, pd.DataFrame):
            self.df = csv_file.reset_index(drop=True)
        else:
            raise AutoMLValidationError("csv_file must be a path or DataFrame")

        self.input_col = input_col
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        if torch.is_tensor(idx):
            idx = idx.item()
        row = self.df.iloc[idx]
        return str(row[self.input_col]), str(row[self.target_col])


class CausalLMFromCSVDataset(Dataset):
    """Dataset for causal language modelling tasks.

    Expected CSV column: ``text``.  The datamodule tokenises and shifts
    labels automatically.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        text_col: str = "text",
    ):
        if isinstance(csv_file, Path):
            try:
                self.df = pd.read_csv(csv_file)
            except FileNotFoundError:
                logger.error("Dataset CSV file not found: %s", csv_file)
                raise
            except pd.errors.EmptyDataError:
                logger.error("Dataset CSV file is empty: %s", csv_file)
                raise AutoMLDataError(f"Dataset CSV file is empty: {csv_file}")
            except pd.errors.ParserError as e:
                logger.error("Failed to parse dataset CSV file: %s", e)
                raise
            except Exception as e:
                logger.error("Unexpected error reading dataset CSV file: %s", e)
                raise
        elif isinstance(csv_file, pd.DataFrame):
            self.df = csv_file.reset_index(drop=True)
        else:
            raise AutoMLValidationError("csv_file must be a path or DataFrame")

        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> str:
        if torch.is_tensor(idx):
            idx = idx.item()
        return str(self.df.iloc[idx][self.text_col])


class MultimodalClassificationDataset(Dataset):
    """Torch dataset for multimodal image classification with auxiliary tabular features.

    In addition to image + label (like ``ImageClassificationFromCSVDataset``),
    this dataset also returns auxiliary feature values from extra CSV columns.
    Raw values are returned here; encoding/scaling is handled by the datamodule.

    Returns ``(PIL.Image, aux_array, int_label)`` per sample.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        root_dir: Path,
        img_col: str = "filename",
        label_col: str = "label",
        auxiliary_columns: list[str] | None = None,
        transform: Optional[T.Compose] = None,
    ):
        if isinstance(csv_file, Path):
            try:
                self.label_csv = pd.read_csv(csv_file)
            except FileNotFoundError:
                logger.error("Dataset CSV file not found: %s", csv_file)
                raise
            except pd.errors.EmptyDataError:
                logger.error("Dataset CSV file is empty: %s", csv_file)
                raise AutoMLDataError(f"Dataset CSV file is empty: {csv_file}")
            except pd.errors.ParserError as e:
                logger.error("Failed to parse dataset CSV file: %s", e)
                raise
            except Exception as e:
                logger.error("Unexpected error reading dataset CSV file: %s", e)
                raise
        elif isinstance(csv_file, pd.DataFrame):
            self.label_csv = csv_file.reset_index(drop=True)
        else:
            raise AutoMLValidationError("csv_file must be a path or DataFrame")

        self.root_dir = root_dir
        self.img_col = img_col
        self.label_col = label_col
        self.auxiliary_columns = auxiliary_columns or []
        self.transform = transform

        if self.label_csv[self.label_col].dtype not in [int, float]:
            self.classes = sorted(self.label_csv[self.label_col].unique().tolist())
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(self.classes)
            }
            self.idx_to_class = {
                idx: cls_name for cls_name, idx in self.class_to_idx.items()
            }
            self.label_csv[self.label_col] = self.label_csv[self.label_col].map(
                self.class_to_idx
            )
        else:
            self.classes = sorted(self.label_csv[self.label_col].unique().tolist())
            self.class_to_idx = {cls: cls for cls in self.classes}
            self.idx_to_class = {cls: cls for cls in self.classes}

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.label_csv.iloc[idx]
        label_idx = int(row[self.label_col])
        label_name = self.idx_to_class[label_idx]

        filename = str(row[self.img_col]).strip()

        img_path = self.root_dir / label_name / filename
        if not img_path.exists():
            logger.error(
                "Image not found: root_dir=%s, label_name=%s, filename=%s",
                self.root_dir,
                label_name,
                filename,
            )
            raise AutoMLDataError(
                f"Image not found\n"
                f"Expected path: {img_path}\n"
                f"root_dir: {self.root_dir}\n"
                f"label_name: {label_name}\n"
                f"filename: {repr(filename)}"
            )

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to open or convert image %s: %s", img_path, e)
            raise

        if self.transform:
            img = self.transform(img)

        if self.auxiliary_columns:
            aux_values = np.array(
                [row[col] for col in self.auxiliary_columns], dtype=np.float32
            )
        else:
            aux_values = np.array([], dtype=np.float32)

        return img, aux_values, torch.tensor(label_idx, dtype=torch.long)
