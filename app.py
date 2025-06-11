import asyncio
import json
import mimetypes
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

import nest_asyncio
import ollama
import pandas as pd
import streamlit as st
from docx import Document
from ollama import AsyncClient as OllamaAsyncClient
from openai import AsyncOpenAI

import src.tasks as tasks
from src.tabular.pipeline import AutoGluonTabularPipeline
from src.tasks import (LLMProcessingTask, TabularSupervisedClassificationTask,
                       TabularSupervisedRegressionTask,
                       TabularSupervisedTimeSeriesTask)

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
    def csv_summary(path: Path) -> str:
        try:
            df = pd.read_csv(path)
            print(df.describe)
            return str(df.describe())
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
                ".ts",
                ".txt",
            ]:
                return file_path.read_text(encoding="utf-8", errors="ignore")
            elif file_path.suffix.lower() == ".docx":
                return FileHandler.read_word_document(file_path)
            elif file_path.suffix.lower() == ".csv":
                return FileHandler.csv_summary(file_path)
            else:
                return f"📦 Binary or unsupported file type: {file_path.name} ({mime})"
        except Exception as e:
            return f"⚠️ Failed to read {file_path.name}: {e}"

    @staticmethod
    def save_temp_file(file) -> tuple[str, str, Path]:
        """Save uploaded file to a temp file and return (filename, mime, tmp_path)."""
        file_suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
            tmp.write(file.read())
            tmp_path = Path(tmp.name)

        mime_type, _ = mimetypes.guess_type(tmp_path.name)
        return file.name, mime_type or "application/octet-stream", tmp_path

    @staticmethod
    def read_each_file(uploaded_files) -> dict[str, str]:
        """Returns {filename: content} for each uploaded file."""
        result = {}
        for file in uploaded_files:
            filename, mime, tmp_path = FileHandler.save_temp_file(file)
            content = FileHandler.read_file_content(tmp_path, mime)
            result[filename] = content
        return result

    @staticmethod
    def aggregate_file_content(uploaded_files) -> tuple[str, dict[str, str]]:
        """Returns an aggregated string summary and file paths."""
        file_info: str = ""
        aggregated_context: str = ""
        file_paths: dict[str, str] = {}

        for file in uploaded_files:
            filename, mime, tmp_path = FileHandler.save_temp_file(file)
            file_suffix = Path(filename).suffix
            file_paths[filename] = str(tmp_path)

            file_info += (
                f"The user has uploaded a file {filename} of type {file_suffix}.\n"
            )
            content = FileHandler.read_file_content(tmp_path, mime)

            aggregated_context += f"\n---\nFile: {filename} ({mime})"
            if file_suffix not in [".zip"]:
                aggregated_context += f"\n{content[:100]}\n"

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

    @staticmethod
    def detect_timestamp_column(user_text: str) -> str:
        response = ChatHandler.chat(
            "Did the user mention what you think could be a timestamp column for the tabular time series data classification/regression? "
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
                    time_stamp_col=file_info["time_stamp_col"],
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
                    print(st.session_state.aggregate_info)
                    task_type = ChatHandler.chat(
                        f"Which task type do you think this is? Answer only with the task type from these options. From the file context, if it is a table or if the user mentions, try to identify if it could be a timeseries task instead of the usual classification/regression. Do not modify the names or add extra spaces: {possible_task_types}. File context {st.session_state.aggregate_info}"
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

                    pipeline = PipelineManager.create_pipeline(
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


class WCAGPipelineHandler:
    @staticmethod
    def process_uploaded_files(uploaded_files):
        try:
            # Use the existing aggregate logic to preserve temp files
            _, file_paths = FileHandler.aggregate_file_content(uploaded_files)

            for path in file_paths:
                SessionStateManager.append_message(
                    "user-hidden", f"The user uploaded a file {path}"
                )
            return file_paths  # Dict[str, str]
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
            files = FileHandler.read_each_file(uploaded_files)  # {filename: content}
        except Exception as e:
            SessionStateManager.append_message(
                "assistant", f"❌ Error processing files: {e}"
            )
            return

        for filename, content in files.items():
            try:
                if st.session_state.get("stop_requested", False):
                    SessionStateManager.append_message(
                        "assistant", "🛑 Processing stopped by user."
                    )
                    return
                chunk_size = 3000  # Adjust based on model/token limits

                # Split the content into lines first
                lines = content.splitlines()
                line_offsets = [0]
                for line in lines:
                    line_offsets.append(
                        line_offsets[-1] + len(line) + 1
                    )  # +1 for newline

                chunks = []
                line_ranges = []

                i = 0
                while i < len(content):
                    end = i + chunk_size
                    chunks.append(content[i:end])

                    # Find corresponding line numbers
                    start_line = (
                        next(j for j, offset in enumerate(line_offsets) if offset > i)
                        - 1
                    )
                    end_line = (
                        next(
                            (
                                j
                                for j, offset in enumerate(line_offsets)
                                if offset > end
                            ),
                            len(lines),
                        )
                        - 1
                    )
                    line_ranges.append(
                        (start_line + 1, end_line + 1)
                    )  # Line numbers are 1-based

                    i = end

                scores = []
                chunk_outputs = []

                SessionStateManager.append_message(
                    "assistant",
                    f"📂 Processing `{filename}` in {len(chunks)} chunk(s)...",
                )

                for i, (chunk, (start_line, end_line)) in enumerate(
                    zip(chunks, line_ranges)
                ):
                    chunk_prompt = f"""
                You're a WCAG (Web Content Accessibility Guidelines) checker. Do not explain the code, your job is to only look at the guidelines.
                Evaluate the following file named `{filename}` and give feedback as follows: If the file contains guidelines, ignore it of course. Do not explain the code.

                1. Score from 0–10 on how well it follows WCAG (0 = not at all, 10 = perfect). This regex will be used to parse it Score[:\\s]*([0-9](?:\\.\\d+)?), response, re.IGNORECASE
                2. If score < 10, list specific improvements needed, with code suggestions in markdown.
                3. Use the WCAG 2.1 guidelines unless a custom guideline file is provided.
                4. Only evaluate the code below. Do not make assumptions beyond the content.
                Always mention the score.

                ### Begin File Content (chunk {i+1} of {len(chunks)}, lines {start_line}–{end_line})
                {chunk}
                ### End File Content
                """

                    with st.spinner(
                        f"🔎 Analyzing `{filename}` - chunk {i+1}/{len(chunks)}..."
                    ):
                        response = ChatHandler.chat(message=chunk_prompt)

                    # Extract score
                    import re

                    score_match = re.search(
                        r"\bScore[:\s]*([0-9](?:\.\d+)?)", response, re.IGNORECASE
                    )
                    score = float(score_match.group(1)) if score_match else None
                    if score is not None:
                        scores.append(score)

                    chunk_feedback = f"""
    📄 **Chunk {i+1}/{len(chunks)} of `{filename}` with approx line {start_line}-{end_line}**  
    {response}
                    """
                    SessionStateManager.append_message("assistant", chunk_feedback)
                    chunk_outputs.append(chunk_feedback)

                # After all chunks processed
                avg_score = round(sum(scores) / len(scores), 2) if scores else "N/A"
                summary = f"✅ Finished analyzing `{filename}`.\n\n**Average WCAG Score:** {avg_score}/10"
                SessionStateManager.append_message("assistant", summary)

            except Exception as e:
                SessionStateManager.append_message(
                    "assistant", f"❌ Failed to process `{filename}`: {e}"
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
                AutoMLTabularPipelineHandler.handle_user_input(prompt, uploaded_files)

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
                WCAGPipelineHandler.handle_user_input(prompt, uploaded_files)
                st.session_state.stop_requested = False

                # Rerender chat after processing
                st.rerun()
    with col2:
        if st.button("🛑 Stop Processing"):
            st.session_state.stop_requested = True
            st.warning("Stop requested. Trying to halt processing...")


if __name__ == "__main__":
    main()
