import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from automl_engine.chat_handler import ChatHandler
from automl_engine.file_handler import FileHandler
from automl_engine.models import Message, SessionState
from automl_engine.pipelines.base import BasePipeline
from automl_engine.pipelines.models import (
    TabularSupervisedClassificationTask, TabularSupervisedRegressionTask,
    TabularSupervisedTimeSeriesTask)
from automl_engine.utils import render_template
from pydantic import BaseModel, FilePath, field_validator


class FileProcessor:
    def __init__(self, session_state: SessionState):
        self.session_state = session_state

    def process_files(self, uploaded_files):
        self.session_state.add_message(
            role="assistant", content="Processing uploaded files"
        )

    def parse_files(self, uploaded_files):
        aggregated_context, file_paths = FileHandler.aggregate_file_content_and_paths(
            uploaded_files
        )
        train_file_path, test_file_path = "", ""

        for key, path in file_paths.items():
            if "train" in key.lower():
                train_file_path = path
                self.session_state.file_info.train_file = path
                self.session_state.add_message(
                    role="assistant", content=f"Found a train file {key}"
                )
            elif "test" in key.lower():
                test_file_path = path
                self.session_state.file_info.test_file = path
                self.session_state.add_message(
                    role="assistant", content=f"Found a test file {key}"
                )

        if not train_file_path:
            self.session_state.add_message(
                role="assistant", content="No train file found"
            )
        if not test_file_path:
            self.session_state.add_message(
                role="assistant", content="No test file found"
            )

        self.session_state.aggregate_info = aggregated_context[:300]
        self.session_state.files_parsed = True

        return train_file_path, test_file_path


class TargetColumnDetector:
    def __init__(self, session_state: SessionState):
        self.session_state = session_state

    def validate(self, train_file, target_col):
        try:
            df = pd.read_csv(train_file)
            return target_col.strip() in df.columns
        except Exception as e:
            print(e)
            return False

    def detect(self, user_text, train_file):
        self.session_state.add_message(
            role="assistant", content="Analyzing your input for target column..."
        )
        query = render_template(
            jinja_environment=self.session_state.jinja_environment,
            template_name="tabular_query_checker.txt",
            user_text=user_text,
            messages=self.session_state.get_all_messages_by_role(["user"]),
        )
        result = ChatHandler.chat(query, context="").strip()

        if result.lower() == "no":
            self.session_state.add_message(
                role="assistant",
                content="❓ I couldn't identify the target column. Please specify which column we should predict.",
            )
            return None

        if not self.validate(train_file, result):
            self.session_state.add_message(
                role="assistant", content=f"Could not find the target column {result}"
            )
            return None

        self.session_state.add_message(
            role="assistant", content=f"Identified target column {result}"
        )
        return result


class TaskTypeDetector:
    def __init__(self, session_state: SessionState, possible_tasks):
        self.session_state = session_state
        self.possible_tasks = possible_tasks

    def detect(self):
        options = ", ".join(task.__name__ for task in self.possible_tasks.values())
        query = render_template(
            jinja_environment=self.session_state.jinja_environment,
            template_name="tabular_task_type_checker.txt",
            task_type_options=options,
        )
        result = ChatHandler.chat(query).strip().lower()
        self.session_state.add_message(
            role="assistant", content=f"Identified task type : {result}..."
        )
        return result


class TimeBudgetDetector:
    def __init__(self, session_state: SessionState):
        self.session_state = session_state

    def detect(self, user_input):
        query = render_template(
            jinja_environment=self.session_state.jinja_environment,
            template_name="tabular_time_checker.txt",
            user_input=user_input,
        )
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


class AutoMLTrainer:
    def __init__(self, session_state: SessionState, save_model_path):
        self.session_state = session_state
        self.save_model_path = save_model_path

    def train(self, train_file, test_file, target_column, time_limit):
        train_data = TabularDataset(pd.read_csv(train_file))
        predictor = TabularPredictor(
            label=target_column, path=self.save_model_path
        ).fit(train_data=train_data, time_limit=time_limit)
        test_data = TabularDataset(pd.read_csv(test_file))
        leaderboard = predictor.leaderboard(test_data)

        if leaderboard is not None:
            self.session_state.add_message(role="assistant", content="Best models")
            self.session_state.add_message(
                role="assistant", content=leaderboard.to_markdown()
            )


class TabularClassificationInput(BaseModel):
    train_csv: FilePath
    test_csv: Optional[FilePath] = None
    target_col: str

    @field_validator("train_csv", "test_csv", mode="before")
    def check_csv_extension(cls, v):
        if v is not None and Path(v).suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got: {v}")
        return v


class AutoMLTabularPipeline(BasePipeline):
    def __init__(self, session_state: SessionState, output_placeholder_ui_element):
        super().__init__(session_state, output_placeholder_ui_element)
        self.detected_col = False
        self.possible_tasks = {
            "supervised classification": TabularSupervisedClassificationTask,
            "supervised regression": TabularSupervisedRegressionTask,
        }
        self.time_limit_for_automl = 10
        self.initial_display_message = render_template(
            jinja_environment=self.session_state.jinja_environment,
            template_name="automl_tabular_initial_message.txt",
        )

        self.session_state.current_model_path = str(
            Path(self.session_state.automloutputpath) / str(time.time())
        )
        self.save_model_path = str(Path(self.session_state.current_model_path))
        os.makedirs(self.save_model_path, exist_ok=True)

        self.file_processor = FileProcessor(session_state)
        self.target_detector = TargetColumnDetector(session_state)
        self.task_detector = TaskTypeDetector(session_state, self.possible_tasks)
        self.time_detector = TimeBudgetDetector(session_state)
        self.trainer = AutoMLTrainer(session_state, self.save_model_path)

    def main_flow(self, user_input: str, uploaded_files):
        if not uploaded_files:
            return

        if not self.session_state.files_parsed:
            self.file_processor.process_files(uploaded_files)
            if self.session_state.stop_requested:
                self.session_state.add_message(
                    role="assistant", content="Processing stopped"
                )
                return
            train_file_path, test_file_path = self.file_processor.parse_files(
                uploaded_files
            )
            self.session_state.train_file_path = train_file_path
            self.session_state.test_file_path = test_file_path

            detected_target = self.target_detector.detect(user_input, train_file_path)
            if detected_target:
                self.session_state.pipeline_state["target_column"] = detected_target
                self.session_state.pipeline_state["stage"] = "detect_task_type"
            else:
                self.session_state.pipeline_state["stage"] = "detect_target_column"
                self.session_state.add_message(
                    role="assistant",
                    content="❓ Please tell me which column you're trying to predict.",
                )
                return

        stage = self.session_state.pipeline_state.get("stage")

        if stage == "detect_target_column":
            target = self.target_detector.detect(
                user_input, self.session_state.train_file_path
            )
            if target:
                self.session_state.pipeline_state["target_column"] = target
                self.session_state.pipeline_state["stage"] = "detect_task_type"

        elif stage == "detect_task_type":
            task_type = self.task_detector.detect()
            if task_type == "supervised time series":
                self.session_state.add_message(
                    role="assistant",
                    content="📉 Time series task detection is not yet supported.",
                )
                return
            self.session_state.pipeline_state["task_type"] = task_type
            self.session_state.pipeline_state["stage"] = "detect_time_limit"
            self.session_state.add_message(
                role="assistant",
                content="⏱️ How long should I run the AutoML training? Eg: 1 minute, or 30 minutes",
            )
            # TODO: this does not move ahead without another message

        elif stage == "detect_time_limit":
            time_limit = self.time_detector.detect(user_input)
            if time_limit:
                self.time_limit_for_automl = time_limit
                self.session_state.pipeline_state["stage"] = "train_and_eval"

        elif stage == "train_and_eval":
            self.trainer.train(
                self.session_state.train_file_path,
                self.session_state.test_file_path,
                self.session_state.pipeline_state["target_column"],
                self.time_limit_for_automl,
            )
