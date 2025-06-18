import mimetypes
import re
import tempfile
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import nest_asyncio
import pandas as pd
import streamlit as st
from docx import Document
from markdown import markdown

nest_asyncio.apply()
from io import BytesIO

import ollama
from pydantic import BaseModel, Field

from src.file_handler import FileHandler


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
            return response["message"]["content"].strip()
        except Exception as e:
            error_message = str(e).lower()
            if "not found" in error_message:
                return f"Model '{model}' not found. Please refer to Documentation at https://ollama.com/library."
            else:
                return f"An unexpected error occurred with model '{model}': {str(e)}"


class FileInfo(BaseModel):
    train_file: Path = Path()
    test_file: Path = Path()
    target_col: Path = Path()
    time_stamp_col: Path = Path()


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class SessionState(BaseModel):
    page_title: str = "Project Assistant"
    messages: List[Message] = Field(default_factory=list)
    files_parsed: bool = False
    stop_requested: bool = False
    aggregate_info: str = ""
    file_info: FileInfo = Field(default_factory=FileInfo)

    def reset(self) -> None:
        """Reset everything"""
        self.messages = []
        self.file_info = FileInfo()
        self.aggregate_info = ""
        self.files_parsed = False
        self.stop_requested = False

    def add_message(self, role: str = "assistant", content: str = ""):
        self.messages.append(Message(role=role, content=content))

    def generate_html_from_session(self) -> BytesIO:
        html_content: str = f"""
        <html>
            <head>
                <meta charset='UTF-8'>
                <title>{self.page_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                    .message {{ margin-bottom: 20px; }}
                    .user {{ font-weight: bold; color: #2a4d9b; }}
                    .assistant {{ font-weight: bold; color: #228B22; }}
                    hr {{ border: none; border-top: 1px solid #ccc; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>{self.page_title}</h1>
        """

        for message in self.messages:
            if message.role != "user-hidden" and len(message.content) > 0:
                role_label: str = message.role.capitalize()
                content_html: str = markdown(message.content)
                html_content += f"""
                    <div class="message">
                        <div class="{message.role}">{role_label}:</div>
                        <div>{content_html}</div>
                    </div>
                    <hr>
                """

        html_content += "</body></html>"

        # Convert to BytesIO for streaming
        html_bytes: BytesIO = BytesIO(html_content.encode("utf-8"))
        html_bytes.seek(0)
        return html_bytes


class build_ui_with_chat:
    def __init__(self, session_state: SessionState) -> None:
        self.session_state: SessionState = session_state
        self.PIPELINES = {
            "-- Select a Pipeline --": None,
            "AutoML Tabular": AutoMLTabularPipeline,
            "WCAG Guidelines": WCAGPipeline,
        }

    def build_ui(self) -> None:
        """Basic page info"""
        st.set_page_config(self.session_state.page_title, layout="wide")
        st.title("ALFIE AutoML Engine")
        st.subheader(
            "Note that the AI can often make mistakes. Before doing anything important, please verify it."
        )
        st.sidebar.header("Extras")

        chat_area = st.container()
        prompt = st.chat_input(
            "What would you like help with? Upload your files for context"
        )
        with chat_area:
            self.output_placeholder = st.empty()
            self.display_messages()
            pipeline_name = st.selectbox(
                "Choose a Pipeline",
                list(self.PIPELINES.keys()),
                key="pipeline_selector",
                index=0,
            )
            uploaded_files = st.file_uploader(
                "📂 Upload files", accept_multiple_files=True, key="file_uploader"
            )

            handler_class = self.PIPELINES.get(pipeline_name)

        if prompt:
            self.session_state.add_message(role="user", content=prompt)
            if handler_class:
                chosen_pipeline_class = handler_class(
                    session_state=self.session_state,
                    output_placeholder_ui_element=self.output_placeholder,
                )
                with st.spinner("Analyzing files"):
                    result = chosen_pipeline_class.main_flow(prompt, uploaded_files)
            st.rerun()

        self.generate_sidebar()

    def generate_sidebar(self) -> None:
        if st.sidebar.button("🧹clear"):
            self.session_state.reset()
            st.rerun()

        if st.sidebar.button("🛑 stop"):
            self.session_state.stop_requested = True
            st.warning("Stop requested. Trying to halt processing...")
            st.rerun()

        if st.sidebar.button("📄 download chat"):
            html_chat = self.session_state.generate_html_from_session()
            try:
                st.download_button(
                    label="📄 Download Conversation as html",
                    data=html_chat,
                    file_name="chat_session.html",
                    mime="text/html",
                )
            except Exception as e:
                st.warning(e)

    def append_message(self, role: str, content: str, display: bool = True) -> None:
        """Add message to list"""
        message: Message = Message(role=role, content=content)
        self.session_state.messages.append(message)

    def get_conversation_text_by_role(
        self, role: str | List[str] = ["user", "user-hidden"]
    ) -> str:
        """Get all user text"""
        if type(role) == str:
            role = [role]
        return "\n".join(
            msg.content for msg in self.session_state.messages if msg.role in role
        )

    def display_messages(self) -> None:
        for msg in self.session_state.messages:
            if msg.role != "user-hidden":
                with st.chat_message(msg.role):
                    st.markdown(msg.content)


class BasePipeline(ABC):
    def __init__(self, session_state, output_placeholder_ui_element) -> None:
        super().__init__()
        self.session_state = session_state
        self.output_placeholder_ui_element = output_placeholder_ui_element

    def main_flow(self, user_input: str, uploaded_files) -> Dict[str, Any] | None: ...

    @staticmethod
    def return_basic_prompt() -> str: ...


class AutoMLTabularPipeline(BasePipeline): ...


class WCAGPipeline(BasePipeline):
    def __init__(self, session_state, output_placeholder_ui_element) -> None:
        super().__init__(session_state, output_placeholder_ui_element)
        self.chunk_outputs: List[str] = []
        self.output_placeholder_ui_element = output_placeholder_ui_element

    @staticmethod
    def return_basic_prompt() -> str:
        return "You're a WCAG (Web Content Accessibility Guidelines) checker. Do not explain the code. Your job is only to evaluate the code against the most recent WCAG guidelines."

    @staticmethod
    def _split_into_chunks(content, chunk_size):
        lines = content.splitlines()
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line) + 1)  # +1 for newline

        chunks, line_ranges = [], []
        i = 0
        while i < len(content):
            end = i + chunk_size
            chunks.append(content[i:end])

            start_line = (
                next(j for j, offset in enumerate(line_offsets) if offset > i) - 1
            )
            end_line = (
                next(
                    (j for j, offset in enumerate(line_offsets) if offset > end),
                    len(lines),
                )
                - 1
            )
            line_ranges.append((start_line + 1, end_line + 1))  # 1-based

            i = end
        return chunks, line_ranges

    def _build_chunk_prompt(self, filename, chunk, idx, total, start_line, end_line):
        return (
            self.return_basic_prompt()
            + f"""
Evaluate the following file named `{filename}`:
1. Score from 0–10 on how well it follows WCAG (0 = not at all, 10 = perfect). Use the format: Score: X (we will extract it using regex).
2. If score < 10, list specific improvements needed, with code suggestions in markdown.
3. Only evaluate the code below. Do not make assumptions beyond the content.

### Begin File Content (chunk {idx+1} of {total}, lines {start_line}–{end_line})
{chunk}
### End File Content
"""
        )

    @staticmethod
    def _extract_wcag_score(response):
        match = re.search(r"\bScore[:\s]*([0-9](?:\.\d+)?)", response, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def process_file(self, filename, content, chunk_size=3000):
        chunks, line_ranges = WCAGPipeline._split_into_chunks(content, chunk_size)
        scores = []

        for i, (chunk, (start_line, end_line)) in enumerate(zip(chunks, line_ranges)):
            prompt = self._build_chunk_prompt(
                filename, chunk, i, len(chunks), start_line, end_line
            )
            response = ChatHandler.chat(prompt)
            score = WCAGPipeline._extract_wcag_score(response)
            if score is not None:
                scores.append(score)

            yield {
                "chunk_index": i,
                "start_line": start_line,
                "end_line": end_line,
                "response": response,
                "score": score,
                "num_chunks": len(chunks),
                "filename": filename,
            }

        # Final summary after all chunks
        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        yield {
            "summary": True,
            "filename": filename,
            "num_chunks": len(chunks),
            "average_score": avg_score,
        }

    def main_flow(self, user_input: str, uploaded_files) -> Dict[str, Any] | None:
        self.session_state.add_message(
            role="assistant", content="Processing uploaded files"
        )
        if self.session_state.stop_requested == True:
            self.session_state.add_message(
                role="assistant", content="Processing stopped"
            )
            return

        try:
            files = FileHandler.read_each_file(uploaded_files)
        except Exception as e:
            self.session_state.add_message(
                role="assistant", content=f"Error reading files: {e}"
            )
            return

        for filename, content in files.items():
            self.session_state.add_message(
                role="assistant", content=f"Processing: {filename}"
            )
            for chunk_result in self.process_file(filename, content):
                if st.session_state.get("stop_requested", False):
                    self.session_state.add_message(
                        role="assistant", content="Processing stopped by user."
                    )
                    return

                if "summary" in chunk_result:
                    summary = f"✅ Finished analyzing `{chunk_result['filename']}`.\n\n**Average WCAG Score:** {chunk_result['average_score']}/10"
                    self.session_state.add_message(role="assistant", content=summary)
                    self.chunk_outputs.append(summary)
                else:
                    feedback = f"""
📄 **Chunk {chunk_result['chunk_index']+1}/{chunk_result['num_chunks']} of `{filename}` (lines {chunk_result['start_line']}-{chunk_result['end_line']})**  
{chunk_result['response']}
"""

                    self.session_state.add_message(role="assistant", content=feedback)
                    self.chunk_outputs.append(feedback)
                    self.output_placeholder_ui_element.markdown(
                        "\n\n---\n\n".join(self.chunk_outputs)
                    )


def main():
    # Persist session state across reruns (Streamlit specific)
    if "session" not in st.session_state:
        st.session_state.session = SessionState()

    session_state = st.session_state.session
    ui_builder = build_ui_with_chat(session_state=session_state)
    ui_builder.build_ui()


if __name__ == "__main__":
    main()
