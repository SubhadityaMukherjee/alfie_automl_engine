import asyncio
import json
import mimetypes
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Union

import nest_asyncio
import ollama
import streamlit as st
from docx import Document
from ollama import AsyncClient as OllamaAsyncClient
from openai import AsyncOpenAI

from src.agent import ChatbotTaskSchema, InteractiveAgent, LLMClient
from src.tasks import (LLMProcessingTask, TabularSupervisedClassificationTask,
                       TabularSupervisedRegressionTask,
                       TabularSupervisedTimeSeriesTask)

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


# def identify_uploaded_type_tabular(uploaded_files):


# automl_tabular_pipeline = OrderedDict(
#     {"file_read": identify_uploaded_type}
# )
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


def get_target_column(uploaded_files, file_paths, max_attempts=3) -> tuple[bool, str]:
    """
    Asks the user for the target column repeatedly until valid or max attempts reached.

    Args:
        uploaded_files: List of uploaded files
        file_paths: Dictionary mapping filenames to temporary file paths
        max_attempts: Maximum number of attempts before giving up

    Returns:
        tuple: (success: bool, target_column: str)
    """
    attempt = 0
    target_col = ""

    while attempt < max_attempts:
        # First try to extract from user messages
        combined_user_messages = "\n".join(
            [
                msg["content"]
                for msg in st.session_state.messages
                if msg["role"] == "user"
            ]
        )
        target_check = chat(
            f"Did the user mention a target column for the tabular data? "
            f"If yes, what is the column name? Only return the name, nothing else. "
            f"If no, return 'no'. User messages: {combined_user_messages}"
        )

        if target_check.lower() != "no":
            target_col = target_check.strip()
            return True, target_col

        # If not found in messages, ask directly
        attempt += 1
        remaining_attempts = max_attempts - attempt + 1
        target_col = st.text_input(
            f"Please specify the target column name (attempt {attempt}/{max_attempts}):",
            key=f"target_col_attempt_{attempt}",
        )

        if not target_col:
            return False, ""

        # Validate the column exists in the data
        if validate_target_column(uploaded_files, file_paths, target_col):
            return True, target_col

        st.warning(f"Column '{target_col}' not found in the data. Please try again.")

    return False, ""


def validate_target_column(uploaded_files, file_paths, target_col) -> bool:
    """
    Validates that the target column exists in the uploaded files.
    """
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            file_path = file_paths[file.name]
            try:
                df = pd.read_csv(file_path)
                if target_col in df.columns:
                    return True
            except Exception:
                continue
    return False


import streamlit as st


def handle_query(user_query, pipeline_choice, uploaded_files, conversation_history):
    """
    Handles user queries with support for multi-step conversations.
    Returns a tuple: (response, needs_more_info, is_complete)
    """
    if pipeline_choice == "AutoML Tabular":
        # Initialize file_info in session state if not exists
        if "file_info" not in st.session_state:
            st.session_state.file_info = {
                "train": "",
                "test": "",
                "target_col": "",
                "file_name_col": "",
            }

        # Process uploaded files
        if uploaded_files:
            _, file_paths = aggregate_file_content(uploaded_files)
            train_files = [f for f in file_paths if "train" in f.lower()]
            test_files = [f for f in file_paths if "test" in f.lower()]

            if train_files:
                st.session_state.file_info["train"] = train_files[0]
            if test_files:
                st.session_state.file_info["test"] = test_files[0]

        # Check for target column in the LATEST user message only
        target_col = ""
        if conversation_history and conversation_history[-1]["role"] == "user":
            target_check = chat(
                f"Did the user mention a target column in the data in this message? "
                f"If yes, extract just the column name. If no, return 'no'. "
                f"User message: {conversation_history[-1]['content']}"
            )
            if target_check.strip().lower() != "no":
                target_col = target_check
                st.session_state.file_info["target_col"] = target_col

        # Determine what's missing
        missing_info = []
        if not st.session_state.file_info["train"]:
            missing_info.append(
                "training data file (should contain 'train' in filename)"
            )
        if not st.session_state.file_info["test"]:
            missing_info.append("test data file (should contain 'test' in filename)")
        if not st.session_state.file_info["target_col"]:
            missing_info.append("target column name")

        if missing_info:
            if len(missing_info) == 1:
                msg = f"❓ To proceed with AutoML Tabular, please provide the {missing_info[0]}."
            else:
                msg = f"❓ To proceed with AutoML Tabular, please provide: {', '.join(missing_info[:-1])} and {missing_info[-1]}."
            return msg, True, False

        # If we have everything
        return (
            f"✅ AutoML Tabular setup complete!\n"
            f"- Training file: {st.session_state.file_info['train']}\n"
            f"- Test file: {st.session_state.file_info['test']}\n"
            f"- Target column: {st.session_state.file_info['target_col']}\n"
            f"How would you like to proceed?",
            False,
            True,
        )

    elif pipeline_choice == "ARIA Guidelines":
        return (
            "📚 Starting ARIA-guidelines-based assistant. How can I help you with accessibility or interface design?",
            False,
            True,
        )

    return "⚠️ Unknown pipeline selected.", False, True


def main():
    st.set_page_config(page_title="Project Assistant", layout="wide")
    st.title("🤖 Interactive Project Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False
    if "current_pipeline" not in st.session_state:
        st.session_state.current_pipeline = None
    if "conversation_complete" not in st.session_state:
        st.session_state.conversation_complete = False

    # Sidebar for history and controls
    with st.sidebar:
        st.header("Conversation History")

        # Display condensed history
        for i, msg in enumerate(st.session_state.messages):
            role = "👤" if msg["role"] == "user" else "🤖"
            with st.container():
                st.caption(
                    f"{role}: {msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}"
                )

        st.divider()

        # Pipeline selection - only enabled when not in middle of conversation
        pipeline_choice = st.selectbox(
            "Choose a Pipeline",
            ["AutoML Tabular", "ARIA Guidelines"],
            key="pipeline_selector",
            disabled=st.session_state.awaiting_response
            and not st.session_state.conversation_complete,
        )

        # File uploader - always enabled
        uploaded_files = st.file_uploader(
            "📂 Upload files", accept_multiple_files=True, key="file_uploader"
        )

        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.awaiting_response = False
            st.session_state.current_pipeline = None
            st.session_state.conversation_complete = False
            if "file_info" in st.session_state:
                del st.session_state.file_info
            st.rerun()

    # Main chat area
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display full messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input - only show if we're not waiting for a response or conversation is complete
        if (
            not st.session_state.awaiting_response
            or st.session_state.conversation_complete
        ):
            if user_input := st.chat_input("What would you like help with?"):
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )
                st.session_state.awaiting_response = True
                st.session_state.current_pipeline = st.session_state.pipeline_selector
                st.session_state.conversation_complete = False
                st.rerun()

    # Handle response generation (after user input)
    if (
        st.session_state.awaiting_response
        and not st.session_state.conversation_complete
    ):
        with col1:
            with st.chat_message("assistant"):
                # Get response
                response, needs_more_info, is_complete = handle_query(
                    st.session_state.messages[-1]["content"],
                    st.session_state.current_pipeline,
                    uploaded_files,
                    st.session_state.messages,
                )

                # Display response
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Update state flags
                st.session_state.awaiting_response = needs_more_info
                st.session_state.conversation_complete = is_complete

        # Rerun to ensure UI updates
        st.rerun()


if __name__ == "__main__":
    main()

# def handle_query(user_query, pipeline_choice, uploaded_files):
#     if pipeline_choice == "AutoML Tabular":
#         file_info = {"train": "", "test": "", "file_name_col": ""}

#         if uploaded_files:
#             # uploaded_files is a list of UploadedFile objects
#             # aggregate_file_content should be able to handle them or you can adjust here
#             _, file_paths = aggregate_file_content(uploaded_files)
#             train_files = [f for f in file_paths if "train" in f.lower()]
#             test_files = [f for f in file_paths if "test" in f.lower()]
#             if train_files:
#                 file_info["train"] = train_files[0]
#             if test_files:
#                 file_info["test"] = test_files[0]

#         target_check = chat(
#             f"Did the user mention a target column in the data? "
#             f"If yes, mention the column. If no, return 'no'. "
#             f"User query: {user_query}"
#         )

#         if target_check.strip().lower() == "no":
#             return "❓ You didn't mention the target column. Could you specify which column you're trying to predict?"
#         else:
#             return f"✅ Detected target column: {target_check}. Proceeding with AutoML Tabular setup..."

#     elif pipeline_choice == "ARIA Guidelines":
#         return "📚 Starting ARIA-guidelines-based assistant. How can I help you with accessibility or interface design?"

#     return "⚠️ Unknown pipeline selected."

# st.set_page_config(page_title="Project Assistant", layout="centered")
# st.title("🤖 Interactive Project Assistant")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # Sidebar widgets outside chat input block
# with st.sidebar:
#     pipeline_choice = st.selectbox("Choose a Pipeline", ["AutoML Tabular", "ARIA Guidelines"])
#     uploaded_files = st.file_uploader("📂 Upload files", accept_multiple_files=True)

# # Display chat messages
# for msg in st.session_state["messages"]:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Chat input
# if user_input := st.chat_input("What would you like help with?"):
#     st.session_state["messages"].append({"role": "user", "content": user_input})

#     with st.chat_message("user"):
#         st.markdown(user_input)

#     with st.chat_message("assistant"):
#         response = handle_query(user_input, pipeline_choice, uploaded_files)
#         st.markdown(response)
#         st.session_state["messages"].append({"role": "assistant", "content": response})


# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# st.set_page_config(page_title="Project Assistant", layout="centered")
# st.title("🤖 Interactive Project Assistant")

# with st.form("user_input_form"):
#     user_query = st.text_area("💬 What would you like help with?", height=150)
#     pipeline_choice = st.selectbox("Choose a Pipeline: ", ["AutoML Tabular","ARIA Guidelines"])
#     uploaded_files = st.file_uploader("📂 Upload files", accept_multiple_files=True)
#     submitted = st.form_submit_button("Submit")

# if submitted:
#     st.session_state['user_query'] = user_query
#     st.session_state['pipeline_choice'] = pipeline_choice
#     st.session_state['uploaded_files'] = uploaded_files

#     # TODO: Add zip file support
#     if st.session_state['pipeline_choice'] == "AutoML Tabular":
#         to_check_dict = {
#             "train": "",
#             "test": "",
#             "file_name_col": ""
#         }
#         if uploaded_files:
#             combined_content, file_paths = aggregate_file_content(uploaded_files=uploaded_files)
#             to_check_dict["train"] = [file for file in file_paths if "train" in file][0]
#             to_check_dict["train"] = [file for file in file_paths if "train" in file][0]

#         # check if class information mentioned
#         check_class = chat(f"Did the user mention a target column in the data, if yes mention the column, if no return the word no. User query - {st.session_state['user_query']}")
#         if check_class == "no":
# ask the user if they know the model class

# print(st.session_state)


# def handle_query(user_query, pipeline_choice, uploaded_files):
#     if pipeline_choice == "AutoML Tabular":
#         file_info = {"train": "", "test": "", "file_name_col": ""}

#         if uploaded_files:
#             _, file_paths = aggregate_file_content(uploaded_files)
#             train_files = [f for f in file_paths if "train" in f.lower()]
#             test_files = [f for f in file_paths if "test" in f.lower()]
#             if train_files:
#                 file_info["train"] = train_files[0]
#             if test_files:
#                 file_info["test"] = test_files[0]

#         # Ask LLM if user mentioned target column
#         target_check = chat(
#             f"Did the user mention a target column in the data? "
#             f"If yes, mention the column. If no, return 'no'. "
#             f"User query: {user_query}"
#         )

#         if target_check.strip().lower() == "no":
#             return "❓ You didn't mention the target column. Could you specify which column you're trying to predict?"
#         else:
#             return f"✅ Detected target column: `{target_check}`. Proceeding with AutoML Tabular setup..."

#     elif pipeline_choice == "ARIA Guidelines":
#         return "📚 Starting ARIA-guidelines-based assistant. How can I help you with accessibility or interface design?"

#     return "⚠️ Unknown pipeline selected."


# st.set_page_config(page_title="Project Assistant", layout="centered")
# st.title("🤖 Interactive Project Assistant")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # -- Chat display
# for msg in st.session_state["messages"]:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # -- Chat input
# if user_input := st.chat_input("What would you like help with?"):
#     # Add user message to history
#     st.session_state["messages"].append({"role": "user", "content": user_input})

#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # Get pipeline choice
#     with st.sidebar:
#         pipeline_choice = st.selectbox(
#             "Choose a Pipeline", ["AutoML Tabular", "ARIA Guidelines"]
#         )
#         uploaded_files = st.file_uploader("📂 Upload files", accept_multiple_files=True)

#     # -- Handle the input
#     with st.chat_message("assistant"):
#         response = handle_query(user_input, pipeline_choice, uploaded_files)
#         st.markdown(response)
#         st.session_state["messages"].append({"role": "assistant", "content": response})
#         print(st.session_state)


# async def fill_missing_fields(
#     llm_client: LLMClient,
#     user_query: str,
#     uploaded_files: Union[list, None],
#     schema_instruction: str
# ) -> dict:
#     file_context, file_paths = aggregate_file_content(uploaded_files)
#     context = f"{schema_instruction}\n\nUser Query: {user_query}\n{file_context}"

#     collected_fields = {
#         "task_type": "",
#         "need_to_train": "",
#         "train_file": "",
#         "test_file": "",
#         "target_column": "",
#         "external_handler": ""
#     }

#     def is_complete(fields):
#         return all(fields.values())

#     async def ask_llm(context, fields):
#         prompt = f"""
#         Based on the following information, fill in the missing fields. Return ONLY a JSON object with the schema:
#         {json.dumps(fields, indent=2)}

#         Context:
#         {context}
#         """
#         response = await llm_client.chat([
#             {"role": "system", "content": "You are a task-oriented assistant."},
#             {"role": "user", "content": prompt}
#         ])
#         try:
#             updated = json.loads(response)
#             return {k: updated.get(k, "") for k in fields}
#         except json.JSONDecodeError:
#             return fields  # Fall back to existing state

#     attempts = 0
#     max_attempts = 5

#     while not is_complete(collected_fields) and attempts < max_attempts:
#         collected_fields = await ask_llm(context, collected_fields)
#         # Append any updates from LLM to context so the model sees prior progress
#         context += f"\nCurrent fields after attempt {attempts + 1}:\n{json.dumps(collected_fields, indent=2)}"
#         attempts += 1

#     if not is_complete(collected_fields):
#         raise RuntimeError("Failed to infer all required fields after several attempts.")

#     # Final file path resolution
#     if collected_fields["train_file"] in file_paths:
#         collected_fields["train_file"] = file_paths[collected_fields["train_file"]]
#     if collected_fields["test_file"] in file_paths:
#         collected_fields["test_file"] = file_paths[collected_fields["test_file"]]

#     return collected_fields

# possible_tasks = [
#     LLMProcessingTask,
#     TabularSupervisedClassificationTask,
#     TabularSupervisedRegressionTask,
#     TabularSupervisedTimeSeriesTask,
# ]

# external_handlers = ["openml"]

# task_types = ", ".join([t.__name__ for t in possible_tasks])
# external_handlers = ", ".join(external_handlers)

# instruction = f"""
# You are a chatbot that needs to understand what the user wants to do. From their query, pick one of the following types of tasks.
# Return the JSON output:
# {{
# "task_type": ..,
# "need_to_train": ..,
# "train_file": ..,
# "test_file": ..,
# "target_column": ..,
# "external_handler": ..
# }}

# Schema: {ChatbotTaskSchema.__pydantic_fields__}

# Task type can be one of the following: {task_types}.
# All results should be of string type. If you do not know the answer, return an empty string.
# """

# def run_async(func, *args, **kwargs):
#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError:
#         loop = None
#     if loop and loop.is_running():
#         # If already running (e.g., Streamlit cloud), create new loop
#         return asyncio.new_event_loop().run_until_complete(func(*args, **kwargs))
#     else:
#         return asyncio.run(func(*args, **kwargs))

# async def fill_missing_fields(
#     llm_client,
#     user_query: str,
#     uploaded_files: Union[list, None],
#     schema_instruction: str
# ):
#     file_context, file_paths = aggregate_file_content(uploaded_files)
#     context = f"{schema_instruction}\n\nUser Query: {user_query}\n{file_context}"

#     collected_fields = {
#         "task_type": "",
#         "need_to_train": "",
#         "train_file": "",
#         "test_file": "",
#         "target_column": "",
#         "external_handler": ""
#     }

#     def is_complete(fields):
#         return all(fields.values())

#     async def ask_llm(context, fields):
#         prompt = f"""
#         Based on the following information, fill in the missing fields. Return ONLY a JSON object with the schema:
#         {json.dumps(fields, indent=2)}

#         Context:
#         {context}
#         """
#         response = await llm_client.chat([
#             {"role": "system", "content": "You are a task-oriented assistant."},
#             {"role": "user", "content": prompt}
#         ])
#         try:
#             updated = json.loads(response)
#             return {k: updated.get(k, "") for k in fields}
#         except json.JSONDecodeError:
#             return fields

#     attempts = 0
#     max_attempts = 5

#     while not is_complete(collected_fields) and attempts < max_attempts:
#         collected_fields = await ask_llm(context, collected_fields)
#         context += f"\nCurrent fields after attempt {attempts + 1}:\n{json.dumps(collected_fields, indent=2)}"
#         attempts += 1

#     # return whatever we have, complete or not
#     return collected_fields, file_paths


# Call our async function
#     try:
#         ollama_client = OllamaAsyncClient()
#         llm_client = LLMClient("ollama", "gemma3:4b", ollama_client)
#         collected_fields, file_paths = run_async(
#             fill_missing_fields,
#             llm_client,
#             user_query,
#             uploaded_files,
#             schema_instruction=instruction
#         )

#         # Identify missing fields
#         missing = [k for k, v in collected_fields.items() if not v]
#         if missing:
#             st.warning("I couldn't infer all fields automatically. Please fill in the missing ones below:")
#             for field in missing:
#                 if "file" in field:
#                     uploaded = st.file_uploader(f"Upload for {field}", key=field)
#                     if uploaded:
#                         # save to temp and update
#                         path = f"/tmp/{uploaded.name}"
#                         with open(path, "wb") as f:
#                             f.write(uploaded.getbuffer())
#                         collected_fields[field] = path
#                 else:
#                     collected_fields[field] = st.text_input(f"{field}", key=field)

#             if st.button("Finalize Fields"):
#                 st.success("✅ All required fields collected:")
#                 st.json(collected_fields)
#         else:
#             st.success("✅ All required fields collected:")
#             st.json(collected_fields)

#     except Exception as e:
#         st.error(f"❌ Could not determine fields: {e}")

# # Cleanup temp files
# for path in st.session_state.get('file_paths', {}).values():
#     try:
#         os.remove(path)
#     except Exception:
#         pass


# Streamlit UI
# st.set_page_config(page_title="Project Assistant", layout="centered")
# st.title("🤖 Interactive Project Assistant")

# with st.form("user_input_form"):
#     user_query = st.text_area("💬 What would you like help with?", height=150)
#     pipeline_choice = st.selectbox("Choose a Pipeline: ", ["ARIA Guidelines", "AutoML Tabular"])
#     uploaded_files = st.file_uploader("📂 Upload files", accept_multiple_files=True)
#     submitted = st.form_submit_button("Submit")

# if submitted:
#     st.session_state['user_query'] = user_query
#     st.session_state['pipeline_choice'] = pipeline_choice
#     st.session_state['uploaded_files'] = uploaded_files
#     print("submitted")

#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         ollama_client = OllamaAsyncClient()
#         llm_client = LLMClient("ollama", "gemma3:4b", ollama_client)
#         final_fields = loop.run_until_complete(
#             fill_missing_fields(llm_client, user_query, uploaded_files, schema_instruction= instruction)
#         )
#         st.success("✅ All required fields collected:")
#         st.json(final_fields)
#     except Exception as e:
#         st.error(f"❌ Could not determine all required fields: {e}")

# for path in st.session_state.get('file_paths', {}).values():
#     try:
#         os.remove(path)
#     except Exception as e:
#         print(f"Failed to delete temp file {path}: {e}")

# st.set_page_config(page_title="Project Assistant", layout="centered")
# st.title("🤖 Interactive Project Assistant")

# with st.form("user_input_form"):
#     user_query = st.text_area("💬 What would you like help with?", height=150)
#     st.session_state['user_query'] = user_query
#     pipeline_choice = st.selectbox("Choose a Pipeline: ", ["ARIA Guidelines", "AutoML Tabular"])
#     st.session_state['pipeline_choice'] = pipeline_choice
#     uploaded_files = st.file_uploader("📂 Upload files", accept_multiple_files=True)
#     st.session_state['uploaded_files'] = uploaded_files

#     submitted = st.form_submit_button("Submit")
# if uploaded_files:
#     try:
#         st.session_state['aggregated_file_content'] = aggregate_file_content(uploaded_files)
#     except Exception as e:
#         print(e)
#         raise FileAggregationFailedException

# print(st.session_state)

# backend_choice = st.selectbox("Choose LLM backend:", ["Ollama", "OpenAI"])
# interface_mode = st.radio("Interface mode:", ["Form Mode", "Chat Mode"])

# for key, default in {
#     "conversation_history": [],
#     "task": None,
#     "missing_fields": [],
#     "user_answers": {},
#     "backend": None,
#     "agent": None
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = default

# if st.session_state.backend != backend_choice or st.session_state.agent is None:
#     if backend_choice == "OpenAI":
#         openai_client = AsyncOpenAI()
#         llm_client = LLMClient("openai", "gpt-4", openai_client)
#     else:
#         ollama_client = OllamaAsyncClient()
#         llm_client = LLMClient("ollama", "gemma3:4b", ollama_client)

#     st.session_state.agent = InteractiveAgent(llm_client)
#     st.session_state.backend = backend_choice

# agent = st.session_state.agent

# # if interface_mode == "Form Mode":
# #     with st.form("user_input_form"):
# #         user_query = st.text_area("💬 What would you like help with?", height=150)
# #         uploaded_files = st.file_uploader("📂 Upload files", accept_multiple_files=True)
# #         submitted = st.form_submit_button("Submit")
# # else:
# uploaded_files = st.file_uploader("📂 Upload files", accept_multiple_files=True)
# user_query = st.chat_input("💬 Ask your assistant")
# submitted = user_query is not None

# # --- Read and Aggregate Uploaded Files ---
# aggregated_context = ""
# if uploaded_files:
#     for file in uploaded_files:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
#             tmp.write(file.read())
#             tmp_path = Path(tmp.name)
#             mime_type, _ = mimetypes.guess_type(tmp_path.name)
#             mime_type = mime_type or "application/octet-stream"
#             content = read_file_content(tmp_path, mime_type)
#             aggregated_context += f"\n---\nFile: {file.name} ({mime_type})\n{content}\n"

# # --- Handle Initial User Query ---
# if submitted and user_query:
#     full_query = user_query + "\n" + aggregated_context

#     async def handle_initial_request():
#         task, history = await agent.get_initial_response(full_query, st.session_state.conversation_history)
#         st.session_state.task = task
#         st.session_state.conversation_history = history
#         st.session_state.missing_fields = task.missing_fields()

#     asyncio.run(handle_initial_request())

# # --- Display Chat History ---
# if interface_mode == "Chat Mode":
#     for msg in st.session_state.conversation_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

# # --- Handle Missing Fields ---
# if st.session_state.task and st.session_state.missing_fields:
#     st.subheader("🔧 Missing Info")
#     prompts = st.session_state.task.field_prompts()

#     for field in st.session_state.missing_fields:
#         if field not in st.session_state.user_answers:
#             st.session_state.user_answers[field] = st.text_input(f"{prompts[field]}", key=f"input_{field}")

#     if st.button("Complete Task"):
#         async def complete_missing_fields():
#             for field, user_input in st.session_state.user_answers.items():
#                 if not user_input.strip():
#                     continue

#                 st.session_state.conversation_history.append({"role": "user", "content": user_input})
#                 followup = st.session_state.conversation_history + [{
#                     "role": "system",
#                     "content": f"You're helping complete the missing field '{field}'. Respond ONLY with the value.",
#                 }]
#                 answer = await agent.client.chat(followup)
#                 answer = answer.strip().replace("```", "").replace("json", "")
#                 try:
#                     parsed = json.loads(answer)
#                     if isinstance(parsed, dict) and field in parsed:
#                         answer = parsed[field]
#                 except json.JSONDecodeError:
#                     pass

#                 setattr(st.session_state.task, field, answer)
#                 st.session_state.conversation_history.append({"role": "assistant", "content": answer})

#             st.session_state.missing_fields = st.session_state.task.missing_fields()

#         asyncio.run(complete_missing_fields())

#         agent.client.chat(st.session_state.conversation_history[-1])

# # --- Final Output ---
# if st.session_state.task and not st.session_state.missing_fields:
#     st.success("✅ Task completed successfully.")
#     st.json(st.session_state.task.dict())
