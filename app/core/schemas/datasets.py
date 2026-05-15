import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from app.core.exceptions import AutoMLDataError, AutoMLValidationError

logger = logging.getLogger(__name__)


class BaseCSVDataset(Dataset, ABC):
    """Base class for all CSV-backed datasets.

    Handles loading from a CSV file path or an existing DataFrame with
    uniform error handling, and provides ``__len__`` / index normalization.
    """

    def __init__(self, csv_file: Union[Path, pd.DataFrame]) -> None:
        self.df: pd.DataFrame = self._load_csv(csv_file)

    @staticmethod
    def _load_csv(csv_file: Union[Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(csv_file, Path):
            try:
                return pd.read_csv(csv_file)
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
            return csv_file.reset_index(drop=True)
        else:
            raise AutoMLValidationError("csv_file must be a Path or DataFrame")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _normalize_idx(idx: int) -> int:
        if torch.is_tensor(idx):
            idx = idx.item()
        return idx

    @abstractmethod
    def __getitem__(self, idx: int):
        ...
