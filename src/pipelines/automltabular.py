import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from src.chat_handler import ChatHandler, Message
from src.file_handler import FileHandler
from src.pipelines.base import BasePipeline, PipelineRegistry
from src.pipelines.task_defs import (
    TabularSupervisedClassificationTask,
    TabularSupervisedRegressionTask,
    TabularSupervisedTimeSeriesTask,
)


@PipelineRegistry.register("AutoML Tabular")
class AutoMLTabularPipeline(BasePipeline):
    def __init__(self, session_state, output_placeholder_ui_element) -> None:
        super().__init__(session_state, output_placeholder_ui_element)
        self.detected_col = False
        self.possible_tasks = {
            "supervised classification": TabularSupervisedClassificationTask,
            "supervised regression": TabularSupervisedRegressionTask,
            # "supervised time series": TabularSupervisedTimeSeriesTask,
        }
        self.time_limit_for_automl = 10  # TODO make this by user input loop
        self.initial_display_message = "Hello, I will help you with your tabular dataset. Please upload your training and testing files in the csv format"
        self.session_state.current_model_path = str(
            Path(self.session_state.automloutputpath) / str(time.time())
        )
        self.save_model_path = str(Path(self.session_state.current_model_path))
        os.makedirs(self.save_model_path, exist_ok=True)

    def validate_target_column(self, train_file: str | Path, target_col: str) -> bool:
        try:
            df = pd.read_csv(train_file)
            return target_col.strip() in df.columns
        except Exception as e:
            print(e)
            return False

    def validate_column_from_train(
        self, train_file_path, detected_target_column
    ) -> bool:
        if self.validate_target_column(train_file_path, detected_target_column):
            self.session_state.add_message(
                role="assistant",
                content=f"Found and validated column {detected_target_column}...",
            )
            return True
        self.session_state.add_message(
            role="assistant",
            content=f"Could not find the target column {detected_target_column}",
        )
        return False

    def get_train_test_files(self, uploaded_files) -> Tuple[str, str, str]:
        aggregated_context, file_paths = FileHandler.aggregate_file_content_and_paths(
            uploaded_files=uploaded_files
        )

        train_file_path: str = ""
        test_file_path: str = ""

        try:
            train_key = next(f for f in file_paths if "train" in f.lower())
            train_file_path = file_paths[train_key]
            self.session_state.file_info.train_file = train_file_path
            self.session_state.add_message(
                role="assistant", content=f"Found a train file {train_key}"
            )
        except StopIteration:
            self.session_state.add_message(
                role="assistant", content="No train file found"
            )

        try:
            test_key = next(f for f in file_paths if "test" in f.lower())
            test_file_path = file_paths[test_key]
            self.session_state.file_info.test_file = test_file_path
            self.session_state.add_message(
                role="assistant", content=f"Found a test file {test_key}"
            )
        except StopIteration:
            self.session_state.add_message(
                role="assistant", content="No test file found"
            )

        return aggregated_context, train_file_path, test_file_path

    def train_and_evaluate_automl_using_autogluon(
        self, train_file_path, test_file_path, detect_target_column
    ):
        train_data = TabularDataset(pd.read_csv(train_file_path))
        predictor = TabularPredictor(
            label=detect_target_column,
            path=self.save_model_path,
        ).fit(train_data=train_data, time_limit=self.time_limit_for_automl)

        test_data = TabularDataset(pd.read_csv(test_file_path))
        leaderboard = predictor.leaderboard(test_data)

        if leaderboard is not None:
            self.session_state.add_message(role="assistant", content="Best models")
            self.session_state.add_message(
                role="assistant", content=leaderboard.to_markdown()
            )

    def process_uploaded_files(self, uploaded_files):
        self.session_state.add_message(
            role="assistant", content="Processing uploaded files"
        )

    def handle_stop(self):
        self.session_state.add_message(role="assistant", content="Processing stopped")
        return

    def handle_file_parsing(self, uploaded_files):
        aggregated_context, train_file_path, test_file_path = self.get_train_test_files(
            uploaded_files
        )
        self.session_state.aggregate_info = aggregated_context[:300]
        self.session_state.files_parsed = True
        return train_file_path, test_file_path

    def detect_target_column(self, user_text: str, train_file_path: str) -> str | None:
        self.session_state.add_message(
            role="assistant", content="Analyzing your input for target column..."
        )

        query = (
            "Did the user mention what you think could be a target column for the tabular data classification/regression? "
            "If yes, what is the column name? Only return the column name, nothing else. ignore the words column and similar in the output. "
            "Eg: Classify X -> X, Classify signature column -> signature, recognize different classes -> no, classify -> no, signature column -> signature "
            f"If no, return 'no'. User messages:\n{user_text} {self.session_state.get_all_messages_by_role(['user'])}"
        )
        result = ChatHandler.chat(query).strip()

        if result.lower() == "no":
            self.session_state.add_message(
                role="assistant",
                content="❓ I couldn't identify the target column. Please specify which column we should predict.",
            )
            return None

        if not self.validate_column_from_train(train_file_path, result):
            return None

        self.session_state.add_message(
            role="assistant",
            content=f"Identified target column {result}",
        )
        self.detected_col = True
        return result

    def detect_task_type(self) -> str:
        task_type_options = ", ".join(
            task.__name__ for task in self.possible_tasks.values()
        )
        query = (
            f"Which task type do you think this is? Answer only with the task type from these options. "
            f"From the file context, if it is a table or if the user mentions, try to identify if it could be a timeseries task instead of the usual classification/regression. "
            f"Do not modify the names or add extra spaces: {task_type_options}."
        )
        result = ChatHandler.chat(query).strip().lower()
        self.session_state.add_message(
            role="assistant", content=f"Identified task type : {result}..."
        )
        return result

    def detect_how_long_the_user_wants_to_wait(self, user_input: str) -> int | None:
        query = f"How long does the user want to wait? Return answer in seconds only, if theysay it in minutes, convert it to seconds. If nothing was found, return 10. User: {user_input}"
        result = ChatHandler.chat(query).strip()
        try:
            val = int(result)
            if val > 0:
                return val
        except:
            pass
        self.session_state.add_message(
            role="assistant",
            content="⏳ How long should I run the AutoML training (e.g., 2 minutes)?",
        )
        return None

    def handle_time_series(self):
        self.session_state.add_message(
            role="assistant",
            content="📉 Time series task detection is not yet supported.",
        )
        return

    def ask_target_column(self):
        self.session_state.add_message(
            role="assistant",
            content="❓ Please tell me which column you're trying to predict.",
        )

    def ask_time_limit(self):
        self.session_state.add_message(
            role="assistant",
            content="⏱️ How long should I run the AutoML training? Eg: 1 minute, or 30 minutes",
        )

    def main_flow(self, user_input: str, uploaded_files) -> None:
        state = self.session_state.pipeline_state

        if not uploaded_files:
            return

        if not self.session_state.files_parsed:
            self.process_uploaded_files(uploaded_files)
            if self.session_state.stop_requested:
                return self.handle_stop()
            
            train_file_path, test_file_path = self.handle_file_parsing(uploaded_files)
            self.session_state.train_file_path = train_file_path
            self.session_state.test_file_path = test_file_path

            # Try auto-detecting target column
            detected_target = self.detect_target_column(
                user_text=user_input, train_file_path=train_file_path
            )
            
            if detected_target:
                state["target_column"] = detected_target
                state["stage"] = "detect_task_type"
            else:
                state["stage"] = "detect_target_column"
                return self.ask_target_column()

        if state["stage"] == "detect_target_column":
            if not state.get("target_column"):
                target = self.detect_target_column(
                    user_input, self.session_state.train_file_path
                )
                if not target:
                    return  # still waiting for valid answer
                state["target_column"] = target
            state["stage"] = "detect_task_type"


        if state["stage"] == "detect_task_type":
            if not state.get("task_type"):
                state["task_type"] = self.detect_task_type()
                if state["task_type"] == "supervised time series":
                    return self.handle_time_series()
            state["stage"] = "detect_time_limit"
            return self.ask_time_limit()

        if state["stage"] == "detect_time_limit":
            time_limit = self.detect_how_long_the_user_wants_to_wait(user_input)
            if time_limit is None:
                return  # still waiting
            self.time_limit_for_automl = time_limit
            state["stage"] = "train_and_eval"

        if state["stage"] == "train_and_eval":
            return self.train_and_evaluate_automl_using_autogluon(
                self.session_state.train_file_path,
                self.session_state.test_file_path,
                state["target_column"],
            )
