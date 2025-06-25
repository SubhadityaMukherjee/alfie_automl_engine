from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel


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

    def __name__(self):
        return "Tabular Supervised Classification Task"


class TabularSupervisedRegressionTask(TabularTask):
    """
    Regression task format
    """

    task_type: str = "regression"

    def __name__(self):
        return "Tabular Supervised Regression Task"


class TabularSupervisedTimeSeriesTask(TabularTask):
    """
    Regression task format
    """

    task_type: str = "time_series"
    time_stamp_col: str = "timestamp"

    def __name__(self):
        return "Tabular Supervised Time Series Task"
