import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import nest_asyncio
import pandas as pd
import streamlit as st

from src.chat_module.handler import ChatHandler
from src.file_processing.reader import FileHandler
from src.llm_task.tasks import LLMProcessingTask
from src.llm_task.wcag import WCAGPipeline
from src.tabular_task.pipeline import AutoGluonTabularPipeline, BaseTabularAutoMLPipeline
from src.tabular_task.tasks import (TabularSupervisedClassificationTask,
                                    TabularSupervisedRegressionTask,
                                    TabularSupervisedTimeSeriesTask)
from src.pipeline_manager import PipelineManager

nest_asyncio.apply()



class SessionStateManager:
    @staticmethod
    def initialize_state():
        st.set_page_config(page_title="Project Assistant", layout="wide")
        st.title("ALFIE AutoML Engine")
        st.subheader(
            "Note that the AI can often make mistakes. Before doing anything important, please verify it."
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "file_info" not in st.session_state:
            st.session_state.file_info = {
                "train": "",
                "test": "",
                "target_col": "",
                "time_stamp_col": "",
            }
        if "files_parsed" not in st.session_state:
            st.session_state.files_parsed = False
        if "stop_requested" not in st.session_state:
            st.session_state.stop_requested = False
        if "aggregate_info" not in st.session_state:
            st.session_state.aggregate_info = ""

    @staticmethod
    def append_message(role: str, content: str, display: bool = True):
        """Append message to session state and optionally display it"""
        st.session_state.messages.append({"role": role, "content": content})
        if display and role != "user-hidden":
            with st.chat_message(role):
                st.markdown(content)

    @staticmethod
    def clear_conversation():
        st.session_state.clear()
        st.rerun()

    @staticmethod
    def get_conversation_text() -> str:
        """Get all user and user-hidden messages as text"""
        return "\n".join(
            msg["content"]
            for msg in st.session_state.messages
            if msg["role"] in ["user", "user-hidden"]
        )


class UIComponents:
    @staticmethod
    def display_sidebar():
        st.sidebar.header("Conversation History")
        if "messages" in st.session_state:
            for msg in st.session_state.messages:
                if msg["role"] in ["user", "assistant"]:  # Skip hidden messages
                    role = "👤" if msg["role"] == "user" else "🤖"
                    content = (
                        msg["content"]
                        if isinstance(msg["content"], str)
                        else str(msg["content"])
                    )
                    st.sidebar.caption(
                        f"{role}: {content[:50]}{'...' if len(content) > 50 else ''}"
                    )
        st.sidebar.divider()

        if st.sidebar.button("Clear Conversation"):
            SessionStateManager.clear_conversation()

    @staticmethod
    def render_chat():
        """Render only new chat messages"""
        if "messages" in st.session_state:
            # Only render messages that haven't been displayed yet
            for msg in st.session_state.messages:
                if msg["role"] != "user-hidden" and not msg.get("displayed", False):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                    msg["displayed"] = True  # Mark as displayed

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

class AutoMLTabularUI:

    @staticmethod
    def process_uploaded_files(uploaded_files):
        try:
            aggregate, file_paths = FileHandler.aggregate_file_content(uploaded_files)
            AutoMLTabularUI._log_uploaded_files(file_paths, aggregate)

            AutoMLTabularUI._store_file_paths(file_paths)
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
    def _log_uploaded_files(file_paths, aggregate):
        for path in file_paths:
            SessionStateManager.append_message(
                "user-hidden",
                f"The user uploaded a file {path} with content like {aggregate}",
            )

    @staticmethod
    def _store_file_paths(file_paths):
        def pick_file(file_list, key, success_msg, fallback_msg, fallback_value=""):
            if file_list:
                st.session_state.file_info[key] = file_paths[file_list[0]]
                SessionStateManager.append_message("assistant", success_msg)
            else:
                st.session_state.file_info[key] = fallback_value
                SessionStateManager.append_message("assistant", fallback_msg)

        train_files = [f for f in file_paths if "train" in f.lower()]
        test_files = [f for f in file_paths if "test" in f.lower()]

        pick_file(
            train_files,
            "train",
            "✅ Found training data file",
            "⚠️ No train.csv file provided",
        )
        pick_file(
            test_files, "test", "✅ Found test data file", "ℹ️ No test file provided"
        )

    @staticmethod
    def handle_user_input(user_input: str, uploaded_files):
        if not uploaded_files:
            return

        SessionStateManager.append_message(
            "assistant", "🔍 Processing uploaded files..."
        )
        AutoMLTabularUI.process_uploaded_files(uploaded_files)

        SessionStateManager.append_message(
            "assistant", "🤔 Analyzing your input for target column..."
        )
        user_text = SessionStateManager.get_conversation_text()
        target_column = ChatHandler.detect_target_column(user_text=user_text)

        if target_column.lower() == "no":
            SessionStateManager.append_message(
                "assistant",
                "❓ I couldn't identify the target column. Please specify which column we should predict.",
            )
            return

        AutoMLTabularUI._handle_target_column(target_column)

    @staticmethod
    def _handle_target_column(target_column):
        SessionStateManager.append_message(
            "assistant", f"🔎 Identified potential target column: {target_column}"
        )
        SessionStateManager.append_message("assistant", "⚙️ Validating target column...")

        train_file = st.session_state.file_info.get("train")
        validated_column = DataValidator.validate_target_column(
            train_file, target_column
        )

        if not validated_column:
            SessionStateManager.append_message(
                "assistant",
                f"⚠️ Column '{target_column}' not found in data. Please specify a valid target column.",
            )
            return

        SessionStateManager.append_message(
            "assistant", f"✅ Target column '{target_column}' validated successfully!"
        )
        st.session_state.file_info["target_col"] = target_column

        AutoMLTabularUI._select_and_run_task_pipeline()

    @staticmethod
    def _select_and_run_task_pipeline():
        task_classes = [
            TabularSupervisedClassificationTask,
            TabularSupervisedRegressionTask,
            TabularSupervisedTimeSeriesTask,
        ]
        task_names = ", ".join(cls.__name__ for cls in task_classes)

        SessionStateManager.append_message(
            "assistant", "🤖 Determining the best task type..."
        )
        task_type = ChatHandler.chat(
            f"Which task type do you think this is? Choose only from: {task_names}. File context: {st.session_state.aggregate_info}"
        ).strip()

        SessionStateManager.append_message(
            "assistant", f"🔧 Creating {task_type} pipeline..."
        )

        if task_type == TabularSupervisedTimeSeriesTask:
            AutoMLTabularUI._handle_timestamp_column()

        pipeline = PipelineManager.create_pipeline(
            task_type, st.session_state.file_info
        )
        AutoMLTabularUI._train_and_evaluate_pipeline(pipeline, task_type)

    @staticmethod
    def _handle_timestamp_column():
        SessionStateManager.append_message(
            "assistant",
            "📊 Files processed successfully. Please tell me about your timestamp column.",
        )
        user_text = SessionStateManager.get_conversation_text()
        timestamp_col = ChatHandler.detect_timestamp_column(user_text=user_text)

        if timestamp_col.lower() == "no":
            SessionStateManager.append_message(
                "assistant",
                "❓ I couldn't identify the time stamp column. Please specify which column we should use.",
            )
            return

        SessionStateManager.append_message(
            "assistant", f"🔎 Identified potential time stamp column: {timestamp_col}"
        )
        validated_column = DataValidator.validate_target_column(
            st.session_state.file_info["train"], timestamp_col
        )
        if validated_column:
            SessionStateManager.append_message(
                "assistant",
                f"✅ Timestamp column '{timestamp_col}' validated successfully!",
            )
            st.session_state.file_info["time_stamp_col"] = timestamp_col
        else:
            SessionStateManager.append_message(
                "assistant",
                f"⚠️ Column '{timestamp_col}' not found in data. Please specify a valid timestamp column.",
            )

    @staticmethod
    def _train_and_evaluate_pipeline(pipeline, task_type):
        if not isinstance(pipeline, AutoGluonTabularPipeline):
            return

        time_limit = AutoMLTabularUI._select_time_limit()
        if time_limit is None:
            return

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
                pipeline.fit(time_limit=time_limit)

        SessionStateManager.append_message(
            "assistant", "📊 Model trained successfully! Evaluating results..."
        )
        leaderboard = pipeline.evaluate()
        if leaderboard is not None:
            SessionStateManager.append_message(
                "assistant", "🏆 Model performance results:"
            )
            SessionStateManager.append_message("assistant", leaderboard.to_markdown())

    @staticmethod
    def _select_time_limit():
        selection = st.selectbox(
            "How fast do you want a model? (Longer might be better)",
            ["20 seconds", "5 min", "10 min", "1 hour"],
        )
        time_map = {
            "20 seconds": 20,
            "5 min": 5 * 60,
            "10 min": 10 * 60,
            "1 hour": 60 * 60,
        }
        return time_map.get(selection)


class WCAGPipelineUI:
    pipeline = WCAGPipeline

    @staticmethod
    def process_uploaded_files(uploaded_files):
        try:
            _, file_paths = FileHandler.aggregate_file_content(uploaded_files)
            for path in file_paths:
                SessionStateManager.append_message(
                    "user-hidden", f"The user uploaded a file {path}"
                )
            return file_paths
        except Exception as e:
            SessionStateManager.append_message(
                "assistant", f"❌ Error processing files: {str(e)}"
            )
            return None

    @staticmethod
    def handle_user_input(user_input: str, uploaded_files):
        if not uploaded_files:
            return

        SessionStateManager.append_message(
            "assistant", "🔍 Processing uploaded files..."
        )

        try:
            files = FileHandler.read_each_file(uploaded_files)
        except Exception as e:
            SessionStateManager.append_message(
                "assistant", f"❌ Error reading files: {e}"
            )
            return

        for filename, content in files.items():
            if st.session_state.get("stop_requested", False):
                SessionStateManager.append_message(
                    "assistant", "🛑 Processing stopped by user."
                )
                return

            WCAGPipelineUI._process_file(filename, content)

    @staticmethod
    def _process_file(filename, content, chunk_size=3000):
        chunks, line_ranges = WCAGPipeline._split_into_chunks(
            content, chunk_size
        )
        scores, outputs = [], []

        SessionStateManager.append_message(
            "assistant", f"📂 Processing `{filename}` in {len(chunks)} chunk(s)..."
        )

        for i, (chunk, (start_line, end_line)) in enumerate(zip(chunks, line_ranges)):
            if st.session_state.get("stop_requested", False):
                SessionStateManager.append_message(
                    "assistant", "🛑 Processing stopped by user."
                )
                return

            prompt = WCAGPipeline._build_chunk_prompt(
                filename, chunk, i, len(chunks), start_line, end_line
            )

            with st.spinner(
                f"🔎 Analyzing `{filename}` - chunk {i+1}/{len(chunks)}..."
            ):
                response = ChatHandler.chat(prompt)

            score = WCAGPipeline._extract_wcag_score(response)
            if score is not None:
                scores.append(score)

            feedback = f"""
📄 **Chunk {i+1}/{len(chunks)} of `{filename}` (lines {start_line}-{end_line})**  
{response}
"""
            SessionStateManager.append_message("assistant", feedback)
            outputs.append(feedback)

        avg_score = round(sum(scores) / len(scores), 2) if scores else "N/A"
        summary = f"✅ Finished analyzing `{filename}`.\n\n**Average WCAG Score:** {avg_score}/10"
        SessionStateManager.append_message("assistant", summary)

    

def main():
    SessionStateManager.initialize_state()
    UIComponents.display_sidebar()

    col1, col2 = st.columns([3, 1])

    with col1:
        # Render existing messages first
        UIComponents.render_chat()

        pipeline_selector = st.selectbox(
            "Choose a Pipeline",
            ["AutoML Tabular", "WCAG Guidelines"],
            key="pipeline_selector",
        )

        if st.session_state["pipeline_selector"] == "AutoML Tabular":
            uploaded_files = st.file_uploader(
                "📂 Upload files", accept_multiple_files=True, key="file_uploader"
            )

            # Chat input at the bottom
            if prompt := st.chat_input(
                "What would you like help with? Upload your files for context"
            ):
                SessionStateManager.append_message("user", prompt)

                # Process the input
                AutoMLTabularUI.handle_user_input(prompt, uploaded_files)

                # Rerender chat after processing
                st.rerun()

        if st.session_state["pipeline_selector"] == "WCAG Guidelines":
            uploaded_files = st.file_uploader(
                "📂 Upload files", accept_multiple_files=True, key="file_uploader"
            )

            # Chat input at the bottom
            if prompt := st.chat_input(
                "What would you like help with? Upload your files for context"
            ):
                SessionStateManager.append_message("user", prompt)

                # Process the input
                WCAGPipelineUI.handle_user_input(prompt, uploaded_files)
                st.session_state.stop_requested = False

                # Rerender chat after processing
                st.rerun()
    with col2:
        if st.button("🛑 Stop Processing"):
            st.session_state.stop_requested = True
            st.warning("Stop requested. Trying to halt processing...")


if __name__ == "__main__":
    main()
