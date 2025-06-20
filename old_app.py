from typing import Any, Dict, List, Optional, Union

import nest_asyncio
import streamlit as st

from old_src.chat_module.handler import ChatHandler
from old_src.file_processing.reader import FileHandler
from old_src.tabular_task.pipeline import (AutoGluonTabularPipeline,
                                           BaseTabularAutoMLPipeline,
                                           DataValidator)
from old_src.tabular_task.tasks import *
from old_src.ui.tabular_automl_ui import AutoMLTabularUI
from old_src.ui.ui_components import GeneralUIComponents, SessionStateManager
from old_src.ui.wcag_pipeline_ui import WCAGPipelineUI

nest_asyncio.apply()
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from nicegui import ui


class AutoMLTabularPipelineHandler:
    @staticmethod
    def process_uploaded_files(uploaded_files):
        try:
            aggregate, file_paths = FileHandler.aggregate_file_content(uploaded_files)

            for path in file_paths:
                SessionStateManager.append_message(
                    "user-hidden",
                    f"The user uploaded a file {path} with content like {aggregate}",
                )

            train_files = [f for f in file_paths if "train" in f.lower()]
            test_files = [f for f in file_paths if "test" in f.lower()]

            if train_files:
                st.session_state.file_info["train"] = file_paths[train_files[0]]
                SessionStateManager.append_message(
                    "assistant", "✅ Found training data file"
                )
            else:
                SessionStateManager.append_message(
                    "assistant", "⚠️ No train.csv file provided"
                )
            if test_files:
                st.session_state.file_info["test"] = file_paths[test_files[0]]
                SessionStateManager.append_message(
                    "assistant", "✅ Found test data file"
                )
            else:
                st.session_state.file_info["test"] = ""
                SessionStateManager.append_message(
                    "assistant", "ℹ️ No test file provided"
                )

            st.session_state.aggregate_info = aggregate[:300]
            st.session_state.files_parsed = True
            SessionStateManager.append_message(
                "assistant",
                "📊 Files processed successfully. Please tell me about your target column.",
            )
        except Exception as e:
            SessionStateManager.append_message(
                "assistant", f"❌ Error processing files: {str(e)}"
            )

    @staticmethod
    def handle_user_input(user_input: str, uploaded_files):
        if uploaded_files:
            SessionStateManager.append_message(
                "assistant", "🔍 Processing uploaded files..."
            )
            AutoMLTabularPipelineHandler.process_uploaded_files(
                uploaded_files=uploaded_files
            )

            user_text = SessionStateManager.get_conversation_text()
            SessionStateManager.append_message(
                "assistant", "🤔 Analyzing your input for target column..."
            )
            target_column = ChatHandler.detect_target_column(user_text=user_text)

            if target_column.lower() == "no":
                SessionStateManager.append_message(
                    "assistant",
                    "❓ I couldn't identify the target column. Please specify which column we should predict.",
                )
            else:
                SessionStateManager.append_message(
                    "assistant",
                    f"🔎 Identified potential target column: {target_column}",
                )
                SessionStateManager.append_message(
                    "assistant", "⚙️ Validating target column..."
                )

                validated_column = DataValidator.validate_target_column(
                    train_file=st.session_state.file_info["train"],
                    target_col=target_column,
                )

                if not validated_column:
                    SessionStateManager.append_message(
                        "assistant",
                        f"⚠️ Column '{target_column}' not found in data. Please specify a valid target column.",
                    )
                else:
                    SessionStateManager.append_message(
                        "assistant",
                        f"✅ Target column '{target_column}' validated successfully!",
                    )
                    st.session_state.file_info["target_col"] = target_column

                    possible_tasks = [
                        TabularSupervisedClassificationTask,
                        TabularSupervisedRegressionTask,
                        TabularSupervisedTimeSeriesTask,
                    ]

                    possible_task_types = ", ".join(
                        [t.__name__ for t in possible_tasks]
                    )
                    SessionStateManager.append_message(
                        "assistant", "🤖 Determining the best task type..."
                    )
                    task_type = ChatHandler.chat(
                        f"Which task type do you think this is? Answer only with the task type from these options. From the file context, if it is a table or if the user mentions, try to identify if it could be a timeseries task instead of the usual classification/regression. Do not modify the names or add extra spaces: {possible_task_types}."
                        # File context {st.session_state.aggregate_info}
                    ).strip()
                    SessionStateManager.append_message(
                        "assistant", f"🔧 Creating {task_type} pipeline..."
                    )
                    if task_type == TabularSupervisedTimeSeriesTask:
                        SessionStateManager.append_message(
                            "assistant",
                            "📊 Files processed successfully. Please tell me about your timestamp column.",
                        )
                        user_text = SessionStateManager.get_conversation_text()
                        timestamp_col = ChatHandler.detect_timestamp_column(
                            user_text=user_text
                        )

                        if timestamp_col.lower() == "no":
                            SessionStateManager.append_message(
                                "assistant",
                                "❓ I couldn't identify the time stamp column. Please specify which column we should predict.",
                            )
                        else:
                            SessionStateManager.append_message(
                                "assistant",
                                f"🔎 Identified potential time stamp column: {timestamp_col}",
                            )

                            validated_column = DataValidator.validate_target_column(
                                train_file=st.session_state.file_info["train"],
                                target_col=timestamp_col,
                            )
                            if not validated_column:
                                SessionStateManager.append_message(
                                    "assistant",
                                    f"⚠️ Column '{timestamp_col}' not found in data. Please specify a valid target column.",
                                )
                            else:
                                SessionStateManager.append_message(
                                    "assistant",
                                    f"✅ Timestamp column '{timestamp_col}' validated successfully!",
                                )
                                st.session_state.file_info["time_stamp_col"] = (
                                    timestamp_col
                                )

                    pipeline = BaseTabularAutoMLPipeline.create_pipeline(
                        task_type, st.session_state.file_info
                    )
                    if type(pipeline) == AutoGluonTabularPipeline:
                        time_limit_selection = st.selectbox(
                            label="How fast do you want a model? (Longer might be better)",
                            options=["20 seconds", "5 min", "10 min", "1 hour"],
                        )
                        if time_limit_selection:
                            time_limit_selection_map = {
                                "20 seconds": 20,
                                "5 min": 5 * 60,
                                "10 min": 10 * 60,
                                "1 hour": 60 * 60,
                            }
                            SessionStateManager.append_message(
                                "assistant",
                                f"⚙️ Identified Task type: {task_type}. Training model... (this may take a few minutes)",
                            )
                            if st.session_state.get("stop_requested", False):
                                SessionStateManager.append_message(
                                    "assistant", "🛑 Processing stopped by user."
                                )
                                return
                            with st.spinner("Training model..."):
                                if not st.session_state.get("stop_requested", False):
                                    pipeline.fit(
                                        time_limit=time_limit_selection_map[
                                            time_limit_selection
                                        ]
                                    )

                            SessionStateManager.append_message(
                                "assistant",
                                "📊 Model trained successfully! Evaluating results...",
                            )
                            leaderboard = pipeline.evaluate()
                            if leaderboard is not None:
                                SessionStateManager.append_message(
                                    "assistant", "🏆 Model performance results:"
                                )
                                SessionStateManager.append_message(
                                    "assistant", leaderboard.to_markdown()
                                )


def main():
    SessionStateManager.initialize_state()
    GeneralUIComponents.display_sidebar()

    col1, col2 = st.columns([3, 1])

    # Map pipeline names to their corresponding UI handler classes
    PIPELINES = {
        "AutoML Tabular": AutoMLTabularUI,
        "WCAG Guidelines": WCAGPipelineUI,
    }

    def render_pipeline_ui(pipeline_label, handler_class):
        uploaded_files = st.file_uploader(
            "📂 Upload files", accept_multiple_files=True, key="file_uploader"
        )

        prompt = st.chat_input(
            "What would you like help with? Upload your files for context"
        )
        if prompt:
            SessionStateManager.append_message("user", prompt)
            handler_class.handle_user_input(prompt, uploaded_files)
            if (
                hasattr(st.session_state, "stop_requested")
                and st.session_state.stop_requested
            ):
                st.session_state.stop_requested = False
            st.rerun()

    with col1:
        GeneralUIComponents.render_chat()

        pipeline_name = st.selectbox(
            "Choose a Pipeline",
            list(PIPELINES.keys()),
            key="pipeline_selector",
        )

        handler_class = PIPELINES.get(pipeline_name)
        if handler_class:
            render_pipeline_ui(pipeline_name, handler_class)

    with col2:
        if st.button("🛑 Stop Processing"):
            st.session_state.stop_requested = True
            st.warning("Stop requested. Trying to halt processing...")


if __name__ == "__main__":
    main()
