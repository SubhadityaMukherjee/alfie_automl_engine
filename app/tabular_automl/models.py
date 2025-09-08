from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel


class TabularTask(BaseModel):
    """Base Pydantic model describing common tabular task inputs."""

    target_feature: str
    time_stamp_col: Optional[pd.DataFrame] = None
    train_file_path: Path
    test_file_path: Optional[Path] = None

    class Config:
        arbitrary_types_allowed = True


class TabularSupervisedClassificationTask(TabularTask):
    """Tabular classification task configuration.

    Typical use-cases: churn prediction, loan approval, disease type, etc.
    """

    task_type: str = "classification"


class TabularSupervisedRegressionTask(TabularTask):
    """Tabular regression task configuration.

    Predicts continuous numeric values (e.g., price, salary, demand).
    """

    task_type: str = "regression"


class TabularSupervisedTimeSeriesTask(TabularTask):
    """Time-series forecasting task configuration for tabular data."""

    task_type: str = "time_series"
    time_stamp_col: str = "timestamp"
