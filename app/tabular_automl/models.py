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
    This pipeline is used when the user wants to classify tabular data into categories or classes. Eg queries: Predict disease type, Classify customer churn, Determine loan approval status, classify tabular, tabular classification. This is the default tabular class
    """

    task_type: str = "classification"

class TabularSupervisedRegressionTask(TabularTask):
    """
    This pipeline is used when the user wants to predict a continuous numeric value from tabular data. Eg queries: Predict house prices, Estimate salary, Forecast sales numbers., tabular regreesion
    """

    task_type: str = "regression"

class TabularSupervisedTimeSeriesTask(TabularTask):
    """
    This pipeline is used when the user wants to make predictions over time from sequential tabular data. Eg queries: Forecast stock prices, Predict electricity consumption, Model time-dependent behavior., tabular time series
    """

    task_type: str = "time_series"
    time_stamp_col: str = "timestamp"
