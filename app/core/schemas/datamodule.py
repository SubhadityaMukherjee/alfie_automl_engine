import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from app.core.exceptions import AutoMLDataError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", 32))
DEFAULT_NUM_WORKERS: int = int(os.getenv("DEFAULT_NUM_WORKERS", 0))
DEFAULT_VAL_SPLIT: float = float(os.getenv("DEFAULT_VAL_SPLIT", 0.2))
DEFAULT_TEST_SPLIT: float = float(os.getenv("DEFAULT_TEST_SPLIT", 0.1))
DEFAULT_IMAGE_CLASSIFIER_HF_ID: str = os.getenv(
    "DEFAULT_IMAGE_CLASSIFIER_HF_ID", "google/vit-base-patch16-224"
)


class BaseDataModule(ABC):
    def __init__(
        self,
        csv_file: Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        shuffle: bool = True,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = DEFAULT_IMAGE_CLASSIFIER_HF_ID,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle: bool = shuffle
        self.val_split: float = val_split
        self.test_split: float = test_split
        self.seed: int = seed
        self.hf_model_id: str = hf_model_id

        self.num_classes: int = 0
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

        logger.info("Initializing %s with CSV: %s", self.__class__.__name__, csv_file)
        self.setup()

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def _collate_fn(self, batch) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        ...

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        ...

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        ...

    @staticmethod
    def _read_csv(csv_file: Path) -> pd.DataFrame:
        try:
            logger.info("Reading dataset from %s", csv_file)
            df: pd.DataFrame = pd.read_csv(csv_file)
        except FileNotFoundError as e:
            logger.error("Dataset file not found: %s", e)
            raise
        except pd.errors.EmptyDataError:
            logger.error("Dataset file is empty: %s", csv_file)
            raise AutoMLDataError(f"Dataset file is empty: {csv_file}")
        except pd.errors.ParserError as e:
            logger.error("Failed to parse dataset CSV: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error reading dataset: %s", e)
            raise
        return df

    @staticmethod
    def _split_df(
        df: pd.DataFrame,
        val_split: float,
        test_split: float,
        seed: int,
        stratify_col: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        stratify_train = df[stratify_col] if stratify_col is not None else None
        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=val_split + test_split,
                stratify=stratify_train,
                random_state=seed,
            )
        except ValueError as e:
            logger.error(
                "Failed to split dataset (insufficient samples or invalid stratification): %s",
                e,
            )
            raise

        stratify_val = temp_df[stratify_col] if stratify_col is not None else None
        try:
            relative_val = val_split / (val_split + test_split)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - relative_val,
                stratify=stratify_val,
                random_state=seed,
            )
        except ValueError as e:
            logger.error("Failed to split validation/test data: %s", e)
            raise

        logger.info(
            "Split completed: train=%d, val=%d, test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return train_df, val_df, test_df

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def _build_label_maps(self, classes: list[str]) -> None:
        self.num_classes = len(classes)
        self.id2label = {i: c for i, c in enumerate(classes)}
        self.label2id = {c: i for i, c in enumerate(classes)}
