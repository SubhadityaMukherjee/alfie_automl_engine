import os
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageClassificationFromCSVDataset(Dataset):
    def __init__(
        self,
        csv_file: Union[str, pd.DataFrame],
        root_dir: str,
        img_col: str = "image",
        label_col: str = "label",
        transform: Optional[Callable[[Image.Image], Any]] = None,
    ) -> None:
        super().__init__()
        if isinstance(csv_file, str):
            self.label_csv = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.label_csv = csv_file.reset_index(drop=True)
        else:
            raise ValueError("data must be either a CSV file path or a pandas DataFrame")

        self.root_dir = root_dir
        self.img_col = img_col
        self.label_col = label_col
        if transform is None:
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transform

        if self.label_csv[self.label_col].dtype not in [int, float]:
            self.classes = sorted(self.label_csv[self.label_col].unique().tolist())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

            # Replace labels with integer indices
            self.label_csv[self.label_col] = self.label_csv[self.label_col].map(self.class_to_idx)
        else:
            # If labels are already numeric, infer classes from unique values
            unique_numeric_classes = sorted(self.label_csv[self.label_col].unique().tolist())
            self.classes = unique_numeric_classes
            self.class_to_idx = {int(c): int(c) for c in unique_numeric_classes}
            self.idx_to_class = {int(c): str(c) for c in unique_numeric_classes}

    def __len__(self) -> int:
        return len(self.label_csv)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.item()

        label = int(self.label_csv.iloc[idx][self.label_col])
        img_path = os.path.join(
            self.root_dir,
            self.idx_to_class[label],
            self.label_csv.iloc[idx][self.img_col],
        )

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


