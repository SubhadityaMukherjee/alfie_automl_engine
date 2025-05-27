import pandas as pd
from pathlib import Path
from typing import Optional
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor
from abc import ABC, abstractmethod

from src.tabular.tasks import (
    SupervisedClassificationTask,
    SupervisedRegressionTask,
    SupervisedTimeSeriesTask,
    TabularChecks,
)


def load_and_validate_df(
    path: Optional[Path], target: str, df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """Load and validate a dataframe from path or directly if provided."""
    return TabularChecks(target_feature=target, csv_file_path=path, df=df)()


class BaseTabularAutoMLPipeline(ABC):
    """Base class for tabular AutoML pipelines."""

    def __init__(self, task):
        self.task = task
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.model = None
        self.supported_tasks: list
        self._check_supported_task()

    def _check_supported_task(self) -> None:
        """Ensure the provided task is supported."""
        for supported_task in self.supported_tasks:
            if isinstance(self.task, supported_task):
                return

        if (
            type(self.task) is SupervisedTimeSeriesTask
            and self.task.time_stamp_col is None
        ):
            raise ValueError(
                "Time series task requires a time_stamp_col to be specified."
            )

        raise ValueError(
            f"Task type '{getattr(self.task, 'task_type', str(type(self.task)))}' is not supported. "
            f"Supported types: {[t.__name__ for t in self.supported_tasks]}"
        )

    def load_data(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Load training and testing data from DataFrame or path."""
        self.train_df = load_and_validate_df(
            path=self.task.train_file_path if train_df is None else None,
            target=self.task.target_feature,
            df=train_df,
        )
        if getattr(self.task, "test_file_path", None) or test_df is not None:
            self.test_df = load_and_validate_df(
                path=self.task.test_file_path if test_df is None else None,
                target=self.task.target_feature,
                df=test_df,
            )

    @abstractmethod
    def fit(self, time_limit: int = 60) -> None:
        """Train the model using the training dataset."""
        pass

    @abstractmethod
    def evaluate(self) -> Optional[pd.DataFrame]:
        """Evaluate the model on the test set and return evaluation results."""
        pass


class AutoGluonTabularPipeline(BaseTabularAutoMLPipeline):
    """AutoML pipeline using AutoGluon."""

    def __init__(self, task):
        self.supported_tasks = [
            SupervisedClassificationTask,
            SupervisedRegressionTask,
            SupervisedTimeSeriesTask,
        ]
        super().__init__(task)
        if isinstance(self.task, SupervisedTimeSeriesTask):
            self.predictor = TimeSeriesPredictor(label=self.task.target_feature)
        self.predictor: Optional[TabularPredictor] = None

    def fit(self, time_limit: int = 60) -> None:
        if self.train_df is None:
            self.load_data()
        train_data = TabularDataset(self.train_df)
        self.predictor = TabularPredictor(label=self.task.target_feature).fit(
            train_data, time_limit=time_limit
        )

    def evaluate(self) -> Optional[pd.DataFrame]:
        if self.test_df is not None and self.predictor is not None:
            test_data = TabularDataset(self.test_df)
            return self.predictor.leaderboard(test_data)
        return None
