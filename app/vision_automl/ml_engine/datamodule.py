from typing import Optional, Dict, Any, List, Tuple, cast

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from transformers import AutoImageProcessor

from .dataset import ImageClassificationFromCSVDataset


class ClassificationData:
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        img_col: str = "filename",
        label_col: str = "label",
        batch_size: int = 32,
        num_workers: int = 0,
        transform=None,
        shuffle: bool = True,
        val_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
        hf_model_id: str = "google/vit-base-patch16-224",
    ) -> None:
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.img_col = img_col
        self.label_col = label_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.hf_model_id = hf_model_id

        self.num_classes: int = 0
        self.train_dataset: Optional[ImageClassificationFromCSVDataset] = None
        self.val_dataset: Optional[ImageClassificationFromCSVDataset] = None
        self.test_dataset: Optional[ImageClassificationFromCSVDataset] = None
        self.processor: Optional[AutoImageProcessor] = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

        self.setup()

    def setup(self) -> None:
        df: pd.DataFrame = pd.read_csv(self.csv_file)

        train_df_raw, temp_df_raw = train_test_split(
            df,
            test_size=self.val_split + self.test_split,
            stratify=df.loc[:, self.label_col],
            random_state=self.seed,
        )
        train_df: pd.DataFrame = cast(pd.DataFrame, train_df_raw)
        temp_df: pd.DataFrame = cast(pd.DataFrame, temp_df_raw)

        relative_val = self.val_split / (self.val_split + self.test_split)
        val_df_raw, test_df_raw = train_test_split(
            temp_df,
            test_size=1 - relative_val,
            stratify=temp_df.loc[:, self.label_col],
            random_state=self.seed,
        )
        val_df: pd.DataFrame = cast(pd.DataFrame, val_df_raw)
        test_df: pd.DataFrame = cast(pd.DataFrame, test_df_raw)

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

        self.num_classes = len(self.train_dataset.classes)
        self.id2label = {i: c for i, c in enumerate(self.train_dataset.classes)}
        self.label2id = {c: i for i, c in enumerate(self.train_dataset.classes)}
        self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)

    def _collate_fn(self, batch: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        images, labels = zip(*batch)
        if self.processor is None:
            raise RuntimeError("Processor not initialized. Call setup() first.")
        processor = cast(Any, self.processor)
        pixel_values = processor(images=list(images), return_tensors="pt").pixel_values
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels, dtype=torch.long)}

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )


