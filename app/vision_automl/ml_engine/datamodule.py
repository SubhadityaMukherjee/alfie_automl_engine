from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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

        self.num_classes: int = 0
        self.train_dataset: Optional[ImageClassificationFromCSVDataset] = None
        self.val_dataset: Optional[ImageClassificationFromCSVDataset] = None
        self.test_dataset: Optional[ImageClassificationFromCSVDataset] = None

        self.setup()

    def setup(self) -> None:
        df = pd.read_csv(self.csv_file)

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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


