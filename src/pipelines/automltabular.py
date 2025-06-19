from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from src.pipelines.base import BasePipeline
from src.file_handler import FileHandler
from src.chat_handler import ChatHandler
from typing import Any, Dict, List, Optional
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor


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
    time_stamp_col: str
    def __name__(self):
        return "Tabular Supervised Time Series Task"
class AutoMLTabularPipeline(BasePipeline):
    def __init__(self, session_state, output_placeholder_ui_element) -> None:
        super().__init__(session_state, output_placeholder_ui_element)

    def validate_target_column(self, train_file: str | Path, target_col: str) -> bool:
        try:
            df = pd.read_csv(train_file)
            return target_col.strip() in df.columns
        except Exception as e:
            print(e)
            return False

    def validate_column_from_train(self, train_file, detect_target_column):
        if self.validate_target_column(train_file, detect_target_column):
            self.session_state.add_message(
                role="assistant",
                content=f"Found and validated column {detect_target_column}...",
            )
            return True
        self.session_state.add_message(
            role="assistant",
            content=f"Could not find the target column {detect_target_column}",
        )
        return False

    def get_train_test_files(self, uploaded_files):
        aggregated_context, file_paths = FileHandler.aggregate_file_content_and_paths(
            uploaded_files=uploaded_files
        )

        train_file, test_file = None, None

        try:
            train_key = next(f for f in file_paths if "train" in f.lower())
            train_file = file_paths[train_key]
            self.session_state.file_info.train_file = train_file
            self.session_state.add_message(
                role="assistant", content=f"Found a train file {train_key}"
            )
        except StopIteration:
            self.session_state.add_message(role="assistant", content="No train file found")

        try:
            test_key = next(f for f in file_paths if "test" in f.lower())
            test_file = file_paths[test_key]
            self.session_state.file_info.test_file = test_file
            self.session_state.add_message(
                role="assistant", content=f"Found a test file {test_key}"
            )
        except StopIteration:
            self.session_state.add_message(role="assistant", content="No test file found")

        return aggregated_context, train_file, test_file

    def main_flow(self, user_input: str, uploaded_files) -> Dict[str, Any] | None:
        if not uploaded_files:
            return

        self.session_state.add_message(role="assistant", content="Processing uploaded files")

        if self.session_state.stop_requested:
            self.session_state.add_message(role="assistant", content="Processing stopped")
            return

        aggregated_context, train_file, test_file = self.get_train_test_files(uploaded_files)
        self.session_state.aggregate_info = aggregated_context[:300]
        self.session_state.files_parsed = True

        user_text = self.session_state.get_all_messages_by_role(["user", "user-hidden"])
        self.session_state.add_message(role="assistant", content="Analyzing your input for target column...")

        self.detected_col = False

        detect_target_column = ChatHandler.chat(
            "Did the user mention what you think could be a target column for the tabular data classification/regression? "
            "If yes, what is the column name? Only return the column name, nothing else. ignore the words column and similar in the output. "
            "Eg: Classify signature column -> signature, recognize different classes -> no, classify -> no, signature column -> signature "
            f"If no, return 'no'. User messages:\n{user_text}"
        ).strip()

        if detect_target_column.lower() == "no":
            self.session_state.add_message(
                role="assistant",
                content="❓ I couldn't identify the target column. Please specify which column we should predict.",
            )
            return

        if not self.validate_column_from_train(train_file, detect_target_column):
            return

        self.session_state.add_message(
            role="assistant", content=f"Identified target column {detect_target_column}"
        )
        self.detected_col = True

        possible_tasks = {
            "supervised classification": TabularSupervisedClassificationTask,
            "supervised regression": TabularSupervisedRegressionTask,
            "supervised time series": TabularSupervisedRegressionTask,
        }

        task_type_options = ", ".join(task.__name__ for task in possible_tasks.values())

        task_type = ChatHandler.chat(
            f"Which task type do you think this is? Answer only with the task type from these options. "
            f"From the file context, if it is a table or if the user mentions, try to identify if it could be a timeseries task instead of the usual classification/regression. "
            f"Do not modify the names or add extra spaces: {task_type_options}."
        ).strip().lower()

        self.session_state.add_message(
            role="assistant", content=f"Identified task type : {task_type}..."
        )

        if task_type == "supervised time series":
            return  # Timeseries logic not handled yet

        train_data = TabularDataset(pd.read_csv(train_file))
        predictor = TabularPredictor(label=detect_target_column, path="autogluon_output").fit(
            train_data=train_data, time_limit=10
        )

        test_data = TabularDataset(pd.read_csv(test_file))
        leaderboard = predictor.leaderboard(test_data)

        if leaderboard is not None:
            self.session_state.add_message(role="assistant", content="Best models")
            self.session_state.add_message(
                role="assistant", content=leaderboard.to_markdown()
            )
