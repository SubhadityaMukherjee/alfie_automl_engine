from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel


class LLMProcessingTask(BaseModel):
    """
    LLM processing format
    """

    input_text: str = ""
    query: str = ""

    class Config:
        arbitrary_types_allowed = True


class TabularTask(BaseModel):
    """
    Tabular task format
    """

    target_feature: str
    time_stamp_col: Optional[pd.DataFrame] = None
    train_file_path: Path
    test_file_path: Optional[Path] = None

    class Config:
        arbitrary_types_allowed = True


class TabularSupervisedClassificationTask(TabularTask):
    """
    Classification task format
    """

    task_type: str = "classification"


class TabularSupervisedRegressionTask(TabularTask):
    """
    Regression task format
    """

    task_type: str = "regression"


class TabularSupervisedTimeSeriesTask(TabularTask):
    """
    Regression task format
    """

    task_type: str = "time_series"
    time_stamp_col: str


class TabularChecks:
    """Utility class for validating tabular data and checking target feature."""

    def __init__(
        self,
        target_feature: str,
        csv_file_path: Optional[Path] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        self.target_feature = target_feature
        self.df = df
        self.csv_file_path = csv_file_path

        if self.df is None and self.csv_file_path is not None:
            self.df = self._load_csv()
        elif self.df is None:
            raise ValueError("Either a dataframe or csv_file_path must be provided.")

    def _load_csv(self) -> Optional[pd.DataFrame]:
        try:
            if self.csv_file_path is not None:
                return pd.read_csv(self.csv_file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file '{self.csv_file_path}': {e}")

    def check_target_exists(self) -> bool:
        if self.df is not None:
            return self.target_feature in self.df.columns
        else:
            return False

    def __call__(self) -> Optional[pd.DataFrame]:
        self.check_target_exists()
        return self.df
