from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from pydantic import BaseModel, FilePath, field_validator
from automl_engine.models import SessionState
from abc import ABC


class BasePipeline(ABC):

    def __init__(self, session_state:SessionState, output_placeholder_ui_element) -> None:
        super().__init__()
        self.session_state = session_state
        self.output_placeholder_ui_element = output_placeholder_ui_element
        self.initial_display_message = "Hi I am here"

    def main_flow(self, user_input: str, uploaded_files) -> Dict[str, Any] | None: ...

    @staticmethod
    def return_basic_prompt() -> str: ...

    @staticmethod
    def get_required_files():
        return {"files": "file_upload_multi"}


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
    @staticmethod
    def get_required_files():
        return {"train_csv": "file_upload_train", "test_csv": "file_upload_test", "target_col": "text"}

class TabularSupervisedRegressionTask(TabularTask):
    """
    This pipeline is used when the user wants to predict a continuous numeric value from tabular data. Eg queries: Predict house prices, Estimate salary, Forecast sales numbers., tabular regreesion
    """

    task_type: str = "regression"
    @staticmethod
    def get_required_files():
        return {"train_csv": "file_upload_train", "test_csv": "file_upload_test", "target_col": "text"}



class TabularSupervisedTimeSeriesTask(TabularTask):
    """
    This pipeline is used when the user wants to make predictions over time from sequential tabular data. Eg queries: Forecast stock prices, Predict electricity consumption, Model time-dependent behavior., tabular time series
    """

    task_type: str = "time_series"
    time_stamp_col: str = "timestamp"

    @staticmethod
    def get_required_files():
        return {"train_csv": "file_upload_train", "test_csv": "file_upload_test", "target_col": "text", "timestamp_col": "text"}


ALLOWED_WEB_FILE_TYPES = {".html", ".htm", ".js", ".css"}
