import os
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ImageClassificationFromCSVDataset(Dataset):
    def __init__(
        self,
        csv_file: Union[str, pd.DataFrame],
        root_dir: str,
        img_col: str = "image",
        label_col: str = "label",
        transform: Optional[T.Compose] = None,
    ):
        if isinstance(csv_file, str):
            self.label_csv = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.label_csv = csv_file.reset_index(drop=True)
        else:
            raise ValueError("csv_file must be a path or DataFrame")

        self.root_dir = root_dir
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform or T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.label_csv[self.label_col].dtype not in [int, float]:
            self.classes = sorted(self.label_csv[self.label_col].unique().tolist())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
            self.label_csv[self.label_col] = self.label_csv[self.label_col].map(self.class_to_idx)
        else:
            self.classes = sorted(self.label_csv[self.label_col].unique().tolist())
            self.class_to_idx = {cls: cls for cls in self.classes}
            self.idx_to_class = {idx: cls for cls in self.classes}

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        label = int(self.label_csv.iloc[idx][self.label_col])
        
        # Handle both flat and hierarchical directory structures
        filename = self.label_csv.iloc[idx][self.img_col]
        label_name = self.idx_to_class[label]
        
        # First try: images are in subdirectories by label (e.g., root_dir/label/filename)
        img_path = os.path.join(self.root_dir, label_name, filename)
        
        # If that doesn't exist, try flat structure (e.g., root_dir/filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root_dir, filename)
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
