import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


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
            self.label_csv = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.label_csv = csv_file.reset_index(drop=True)
        else:
            raise ValueError("csv_file must be a path or DataFrame")

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
            print(os.listdir(self.root_dir))
            print(os.listdir(self.root_dir / label_name))

            raise FileNotFoundError(
                f"Image not found\n"
                f"Expected path: {img_path}\n"
                f"root_dir: {self.root_dir}\n"
                f"label_name: {label_name}\n"
                f"filename: {repr(filename)}"
            )

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label_idx, dtype=torch.long)
