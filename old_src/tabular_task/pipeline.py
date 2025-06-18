from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor

import old_src.tabular_task.tasks as tabular_task_modules
from old_src.chat_module.handler import ChatHandler
from old_src.file_processing.reader import FileHandler

from .tasks import (
    TabularChecks,
    TabularSupervisedClassificationTask,
    TabularSupervisedRegressionTask,
    TabularSupervisedTimeSeriesTask,
)


def load_and_validate_df(
    path: Optional[Path], target: str, df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """Load and validate a dataframe from path or directly if provided."""
    return TabularChecks(target_feature=target, csv_file_path=path, df=df)()


class DataValidator:
    @staticmethod
    def validate_target_column(train_file: str | Path, target_col: str) -> bool:
        target_col = target_col.strip()
        try:
            df = pd.read_csv(train_file)
        except Exception as e:
            print(e)
            return False
        return target_col in df.columns


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
            type(self.task) is TabularSupervisedTimeSeriesTask
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

    @staticmethod
    def process_uploaded_files(uploaded_files):
        # TODO Make this more general
        aggregate, file_paths = FileHandler.aggregate_file_content(uploaded_files)
        train_files = [f for f in file_paths if "train" in f.lower()]
        test_files = [f for f in file_paths if "test" in f.lower()]

        file_info = {
            "train": train_files[0] if train_files else "",
            "test": test_files[0] if test_files else "",
        }
        print("finfo", file_info)

        return aggregate, file_paths, file_info

    @staticmethod
    def detect_target_column(conversation_text):
        return ChatHandler.detect_target_column(user_text=conversation_text)

    @staticmethod
    def detect_timestamp_column(conversation_text):
        return ChatHandler.detect_timestamp_column(user_text=conversation_text)

    @staticmethod
    def validate_column(filepath, column_name):
        return DataValidator.validate_target_column(filepath, column_name)

    @staticmethod
    def infer_task_type(aggregate_info):
        task_classes = [
            TabularSupervisedClassificationTask,
            TabularSupervisedRegressionTask,
            TabularSupervisedTimeSeriesTask,
        ]
        task_names = ", ".join(cls.__name__ for cls in task_classes)
        return ChatHandler.chat(
            f"Which task type do you think this is? Choose only from: {task_names}. File context: {aggregate_info}"
        ).strip()

    @staticmethod
    def create_pipeline(task_type, file_info):
        task_class = getattr(tabular_task_modules, task_type)
        try:
            if task_class in [
                TabularSupervisedClassificationTask,
                TabularSupervisedRegressionTask,
                TabularSupervisedTimeSeriesTask,
            ]:
                task = task_class(
                    target_feature=file_info["target_col"],
                    train_file_path=Path(file_info["train"]),
                    test_file_path=Path(file_info["test"]),
                    time_stamp_col=file_info["time_stamp_col"],
                )
                pipeline = AutoGluonTabularPipeline(task, save_path="autogluon_output")
                return pipeline
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return None

    @staticmethod
    def train_pipeline(pipeline, time_limit):
        pipeline.fit(time_limit=time_limit)

    @staticmethod
    def evaluate_pipeline(pipeline):
        return pipeline.evaluate()


class AutoGluonTabularPipeline(BaseTabularAutoMLPipeline):
    """AutoML pipeline using AutoGluon."""

    def __init__(self, task, save_path: str = "autogluon_model"):
        self.supported_tasks = [
            TabularSupervisedClassificationTask,
            TabularSupervisedRegressionTask,
            TabularSupervisedTimeSeriesTask,
        ]
        super().__init__(task)
        self.save_path = save_path
        if isinstance(self.task, TabularSupervisedTimeSeriesTask):
            self.predictor = TimeSeriesPredictor(
                label=self.task.target_feature, path=save_path
            )
        else:
            self.predictor: Optional[TabularPredictor] = None

    def fit(self, time_limit: int = 60) -> None:
        if self.train_df is None:
            self.load_data()
        train_data = TabularDataset(self.train_df)

        self.predictor = TabularPredictor(
            label=self.task.target_feature, path=self.save_path
        ).fit(train_data, time_limit=time_limit)

    def evaluate(self) -> Optional[pd.DataFrame]:
        if self.test_df is not None and self.predictor is not None:
            test_data = TabularDataset(self.test_df)
            return self.predictor.leaderboard(test_data)
        return None
