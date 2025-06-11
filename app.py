import asyncio
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Union

import pandas as pd
import streamlit as st
from docx import Document
from openai import AsyncOpenAI
import nest_asyncio
from ollama import AsyncClient as OllamaAsyncClient

# from src.agent import InteractiveAgent, LLMClient, ChatbotTaskSchema
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


def read_word_document(path: Path) -> str:
    try:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"⚠️ Error reading document: {e}"


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
            return read_word_document(file_path)
        else:
            return f"📦 Binary or unsupported file type: {file_path.name} ({mime})"
    except Exception as e:
        return f"⚠️ Failed to read {file_path.name}: {e}"


class FileAggregationFailedException(Exception):
    def __init__(self):
        self.message = "Failed to aggregate files, check your uploaded files"
        super().__init__(self.message)


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

        # Store file path
        file_paths[file.name] = str(tmp_path)

        mime_type, _ = mimetypes.guess_type(tmp_path.name)
        mime_type = mime_type or "application/octet-stream"
        content = read_file_content(tmp_path, mime_type)
        aggregated_context += f"\n---\nFile: {file.name} ({mime_type})"
        if file_suffix not in [".csv"]:
            aggregated_context += f"\n{content}\n"

    return file_info + aggregated_context, file_paths


def chat(message, model="gemma3:4b"):
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
            return f"Model '{model}' not found. Please refer to Doumentation at https://ollama.com/library."
        else:
            return f"An unexpected error occurred with model '{model}': {str(e)}"


def validate_target_column(train_file: str | Path, target_col: str) -> bool:
    """
    Validates that the target column exists in the uploaded files.
    """
    try:
        df = pd.read_csv(train_file)
    except Exception as e:
        print(e)
        return False
    return target_col in df.columns


def display_sidebar():
    st.sidebar.header("Conversation History")
    for msg in st.session_state.messages:
        role = "👤" if msg["role"] == "user" else "🤖"
        st.sidebar.caption(
            f"{role}: {msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}"
        )

    st.sidebar.divider()

    st.sidebar.selectbox(
        "Choose a Pipeline",
        ["AutoML Tabular", "ARIA Guidelines"],
        key="pipeline_selector",
    )

    uploaded_files = st.sidebar.file_uploader(
        "📂 Upload files", accept_multiple_files=True, key="file_uploader"
    )

    if st.sidebar.button("Clear Conversation"):
        st.session_state.clear()
        st.rerun()

    return uploaded_files


def initialize_state():
    st.set_page_config(page_title="Project Assistant", layout="wide")
    st.title("🤖 Interactive Project Assistant")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault(
        "file_info", {"train": "", "test": "", "target_col": ""}
    )
    st.session_state.setdefault("files_parsed", False)


def append_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def process_uploaded_files(uploaded_files):
    _, file_paths = aggregate_file_content(uploaded_files)

    for path in file_paths:
        append_message("user-hidden", f"The user uploaded a file {path}")

    train_files = [f for f in file_paths if "train" in f.lower()]
    test_files = [f for f in file_paths if "test" in f.lower()]

    if train_files:
        st.session_state.file_info["train"] = file_paths[train_files[0]]
    else:
        append_message("assistant", "No train.csv file provided")
    if test_files:
        st.session_state.file_info["test"] = file_paths[test_files[0]]

    st.session_state.files_parsed = True
    append_message(
        "assistant", "Files processed. You can now ask about the target column, etc."
    )


def detect_target_column():
    user_text = "\n".join(
        msg["content"]
        for msg in st.session_state.messages
        if msg["role"] in ["user", "user-hidden"]
    )

    target_check = chat(
        "Did the user mention what you think could be a target column for the tabular data classification/regression? "
        "If yes, what is the column name? Only return the column name, nothing else. ignore the words column and similar in the output"
        "If no, return 'no'. User messages:\n" + user_text
    )

    append_message("assistant", f"Target column identified: {target_check}")

    return target_check.strip()


def render_chat():
    for msg in st.session_state.messages:
        if msg["role"] != "user-hidden":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


def main():
    initialize_state()
    uploaded_files = display_sidebar()

    col1, col2 = st.columns([3, 1])

    with col1:
        # Render chat history in main window
        render_chat()

        # Input from user
        user_input = st.chat_input("What would you like help with?")
        if user_input:
            append_message("user", user_input)

        if st.session_state["pipeline_selector"] == "AutoML Tabular" and uploaded_files:
            process_uploaded_files(uploaded_files)

            if user_input and uploaded_files:
                with st.chat_message("assistant"):
                    target_col = detect_target_column()

                if target_col.lower() != "no":
                    valid = validate_target_column(
                        train_file=st.session_state.file_info["train"],
                        target_col=target_col,
                    )

                    if valid:
                        append_message(
                            "assistant", f"Target column `{target_col}` is valid."
                        )
                        st.session_state.file_info["target_col"] = target_col

                        possible_tasks = [
                            LLMProcessingTask,
                            TabularSupervisedClassificationTask,
                            TabularSupervisedRegressionTask,
                            TabularSupervisedTimeSeriesTask,
                        ]

                        possible_task_types = ", ".join(
                            [t.__name__ for t in possible_tasks]
                        )

                        # AutoML part
                        task_type = chat(
                            f"Which task type do you think this is? Answer only with the task type - {possible_task_types}"
                        ).strip()
                        try:
                            task_type = getattr(tasks, task_type)
                            append_message(
                                "assistant", f"Identified Task type - {task_type}"
                            )
                            task = TabularSupervisedClassificationTask(
                                target_feature=st.session_state.file_info["target_col"],
                                train_file_path=Path(
                                    st.session_state.file_info["train"]
                                ),
                                test_file_path=Path(st.session_state.file_info["test"]),
                            )

                            pipeline = AutoGluonTabularPipeline(task)
                            append_message("assistant", f"Created Pipeline")
                            pipeline.fit(time_limit=20)
                            leaderboard = pipeline.evaluate()
                            if leaderboard is not None:
                                st.markdown(leaderboard.to_markdown())
                        except Exception as e:
                            print(e)
                            append_message("assistant", f"Could not identify task type")

                    else:
                        append_message(
                            "assistant",
                            f"⚠️ Target column `{target_col}` not found in training data.",
                        )


if __name__ == "__main__":
    main()
