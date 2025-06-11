import asyncio
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, List

import pandas as pd
import streamlit as st
from docx import Document
from openai import AsyncOpenAI
import nest_asyncio
from ollama import AsyncClient as OllamaAsyncClient

import src.tasks as tasks
from src.tasks import (
    LLMProcessingTask,
    TabularSupervisedClassificationTask,
    TabularSupervisedRegressionTask,
    TabularSupervisedTimeSeriesTask,
)
from src.tabular.pipeline import AutoGluonTabularPipeline
from collections import OrderedDict
import ollama

nest_asyncio.apply()


class FileHandler:
    @staticmethod
    def read_word_document(path: Path) -> str:
        try:
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            return f"⚠️ Error reading document: {e}"

    @staticmethod
    def read_file_content(file_path: Path, mime: str) -> str:
        try:
            if mime.startswith("text/") or file_path.suffix.lower() in [
                ".html",
                ".css",
                ".py",
                ".json",
                ".csv",
            ]:
                return file_path.read_text(encoding="utf-8", errors="ignore")
            elif file_path.suffix.lower() == ".docx":
                return FileHandler.read_word_document(file_path)
            else:
                return f"📦 Binary or unsupported file type: {file_path.name} ({mime})"
        except Exception as e:
            return f"⚠️ Failed to read {file_path.name}: {e}"

    @staticmethod
    def aggregate_file_content(uploaded_files) -> tuple[str, dict[str, str]]:
        file_info: str = ""
        aggregated_context: str = ""
        file_paths: dict[str, str] = {}

        for file in uploaded_files:
            file_suffix = Path(file.name).suffix
            file_info += (
                f"The user has uploaded a file {file.name} of type {file_suffix}.\n"
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
                tmp.write(file.read())
                tmp_path = Path(tmp.name)

            file_paths[file.name] = str(tmp_path)

            mime_type, _ = mimetypes.guess_type(tmp_path.name)
            mime_type = mime_type or "application/octet-stream"
            content = FileHandler.read_file_content(tmp_path, mime_type)
            aggregated_context += f"\n---\nFile: {file.name} ({mime_type})"
            if file_suffix not in [".csv"]:
                aggregated_context += f"\n{content}\n"

        return file_info + aggregated_context, file_paths


class ChatHandler:
    @staticmethod
    def chat(message: str, model: str = "gemma3:4b") -> str:
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            error_message = str(e).lower()
            if "not found" in error_message:
                return f"Model '{model}' not found. Please refer to Documentation at https://ollama.com/library."
            else:
                return f"An unexpected error occurred with model '{model}': {str(e)}"

    @staticmethod
    def detect_target_column(user_text: str) -> str:
        response = ChatHandler.chat(
            "Did the user mention what you think could be a target column for the tabular data classification/regression? "
            "If yes, what is the column name? Only return the column name, nothing else. ignore the words column and similar in the output"
            "Eg: Classify signature column -> signature, recognize different classes -> no, classify -> no, signature column -> signature"
            "If no, return 'no'. User messages:\n" + user_text
        )
        return response.strip()


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


class PipelineManager:
    @staticmethod
    def create_pipeline(
        task_type: str, file_info: dict
    ) -> Optional[AutoGluonTabularPipeline]:
        try:
            task_class = getattr(tasks, task_type)

            if task_class in [
                TabularSupervisedClassificationTask,
                TabularSupervisedRegressionTask,
                TabularSupervisedTimeSeriesTask,
            ]:
                task = task_class(
                    target_feature=file_info["target_col"],
                    train_file_path=Path(file_info["train"]),
                    test_file_path=Path(file_info["test"]),
                )
                pipeline = AutoGluonTabularPipeline(task, save_path="autogluon_output")
                return pipeline
            elif task_class == LLMProcessingTask:
                # Handle LLM specific pipeline creation
                pass

        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return None


class SessionStateManager:
    @staticmethod
    def initialize_state():
        st.set_page_config(page_title="Project Assistant", layout="wide")
        st.title("🤖 Interactive Project Assistant")

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "file_info" not in st.session_state:
            st.session_state.file_info = {"train": "", "test": "", "target_col": ""}
        if "files_parsed" not in st.session_state:
            st.session_state.files_parsed = False

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


class AutoMLTabularPipelineHandler:
    @staticmethod
    def process_uploaded_files(uploaded_files):
        try:
            _, file_paths = FileHandler.aggregate_file_content(uploaded_files)

            for path in file_paths:
                SessionStateManager.append_message(
                    "user-hidden", f"The user uploaded a file {path}"
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
                        f"Which task type do you think this is? Answer only with the task type from these options. Do not modify the names or add extra spaces: {possible_task_types}"
                    ).strip()
                    # print(task_type)
                    SessionStateManager.append_message(
                        "assistant", f"🔧 Creating {task_type} pipeline..."
                    )

                    pipeline = PipelineManager.create_pipeline(
                        task_type, st.session_state.file_info
                    )
                    if type(pipeline) == AutoGluonTabularPipeline:
                        SessionStateManager.append_message(
                            "assistant",
                            f"⚙️ Identified Task type: {task_type}. Training model... (this may take a few minutes)",
                        )
                        with st.spinner("Training model..."):
                            pipeline.fit(time_limit=20)

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
    UIComponents.display_sidebar()

    col1, col2 = st.columns([3, 1])

    with col1:
        # Render existing messages first
        UIComponents.render_chat()

        pipeline_selector = st.selectbox(
            "Choose a Pipeline",
            ["AutoML Tabular", "ARIA Guidelines"],
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
                AutoMLTabularPipelineHandler.handle_user_input(prompt, uploaded_files)

                # Rerender chat after processing
                st.rerun()
        
        if st.session_state["pipeline_selector"] == "ARIA Guidelines":
            uploaded_files = st.file_uploader(
                "📂 Upload files", accept_multiple_files=True, key="file_uploader"
            )

            # Chat input at the bottom
            if prompt := st.chat_input(
                "What would you like help with? Upload your files for context"
            ):
                SessionStateManager.append_message("user", prompt)

                # Process the input
                # AutoMLTabularPipelineHandler.handle_user_input(prompt, uploaded_files)

                # Rerender chat after processing
                st.rerun()


if __name__ == "__main__":
    main()
