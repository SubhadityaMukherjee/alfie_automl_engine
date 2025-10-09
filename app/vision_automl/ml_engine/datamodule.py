import os
import logging
from typing import Any, Callable

import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from .dataset import ImageClassificationFromCSVDataset


# --- Environment Setup ---
load_dotenv(find_dotenv())

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Defaults from Environment ---
DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", 32))
DEFAULT_NUM_WORKERS: int = int(os.getenv("DEFAULT_NUM_WORKERS", 0))
DEFAULT_VAL_SPLIT: float = float(os.getenv("DEFAULT_VAL_SPLIT", 0.2))
DEFAULT_TEST_SPLIT: float = float(os.getenv("DEFAULT_TEST_SPLIT", 0.1))
DEFAULT_IMAGE_CLASSIFIER_HF_ID: str = os.getenv(
    "DEFAULT_IMAGE_CLASSIFIER_HF_ID", "google/vit-base-patch16-224"
)


class ClassificationData:
    """Handles dataset preparation and dataloaders for image classification tasks."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        img_col: str = "filename",
        label_col: str = "label",
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        transform: Callable | None = None,
        shuffle: bool = True,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = DEFAULT_IMAGE_CLASSIFIER_HF_ID,
    ) -> None:
        self.csv_file: str = csv_file
        self.root_dir: str = root_dir
        self.img_col: str = img_col
        self.label_col: str = label_col
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.transform: Callable | None = transform
        self.shuffle: bool = shuffle
        self.val_split: float = val_split
        self.test_split: float = test_split
        self.seed: int = seed
        self.hf_model_id: str = hf_model_id

        self.num_classes: int = 0
        self.train_dataset: ImageClassificationFromCSVDataset | None = None
        self.val_dataset: ImageClassificationFromCSVDataset | None = None
        self.test_dataset: ImageClassificationFromCSVDataset | None = None
        self.processor: AutoImageProcessor | None = None
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

        logger.info("Initializing ClassificationData with CSV: %s", csv_file)
        self.setup()

    def setup(self) -> None:
        """Create train/val/test splits, datasets, label maps, and processor."""
        logger.info("Reading dataset from %s", self.csv_file)
        df: pd.DataFrame = pd.read_csv(self.csv_file)

        logger.debug("Initial dataset shape: %s", df.shape)

        # --- Split train, val, test ---
        logger.info("Splitting dataset into train/val/test sets...")
        train_df, temp_df = train_test_split(
            df,
            test_size=self.val_split + self.test_split,
            stratify=df[self.label_col],
            random_state=self.seed,
        )

        relative_val = self.val_split / (self.val_split + self.test_split)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - relative_val,
            stratify=temp_df[self.label_col],
            random_state=self.seed,
        )

        logger.info(
            "Split completed: train=%d, val=%d, test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # --- Build datasets ---
        self.train_dataset = ImageClassificationFromCSVDataset(
            csv_file=train_df,
            root_dir=self.root_dir,
            img_col=self.img_col,
            label_col=self.label_col,
            transform=self.transform,
        )
        self.val_dataset = ImageClassificationFromCSVDataset(
            csv_file=val_df,
            root_dir=self.root_dir,
            img_col=self.img_col,
            label_col=self.label_col,
            transform=self.transform,
        )
        self.test_dataset = ImageClassificationFromCSVDataset(
            csv_file=test_df,
            root_dir=self.root_dir,
            img_col=self.img_col,
            label_col=self.label_col,
            transform=self.transform,
        )

        # --- Label mapping ---
        self.num_classes = len(self.train_dataset.classes)
        self.id2label = {i: c for i, c in enumerate(self.train_dataset.classes)}
        self.label2id = {c: i for i, c in enumerate(self.train_dataset.classes)}

        logger.info("Number of classes detected: %d", self.num_classes)
        logger.debug("Label mappings: %s", self.label2id)

        # --- Processor ---
        self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        logger.info("Loaded processor from: %s", self.hf_model_id)

    def _collate_fn(self, batch: list[tuple[Any, Any]]) -> dict[str, torch.Tensor]:
        """Collate batch using HF processor to produce pixel values and labels."""
        images, labels = zip(*batch)
        if self.processor is None:
            logger.error("Processor not initialized. Call setup() first.")
            raise RuntimeError("Processor not initialized. Call setup() first.")

        pixel_values = self.processor(images=list(images), return_tensors="pt").pixel_values
        logger.debug("Collated batch with %d samples", len(labels))

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        logger.info("Creating training dataloader (batch_size=%d)", self.batch_size)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        logger.info("Creating validation dataloader (batch_size=%d)", self.batch_size)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        logger.info("Creating test dataloader (batch_size=%d)", self.batch_size)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

