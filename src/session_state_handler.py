from typing import List

import nest_asyncio
from markdown import markdown

nest_asyncio.apply()
from datetime import datetime
from io import BytesIO
from pathlib import Path

from pydantic import BaseModel, Field

from src import render_template
from src.chat_handler import Message
from src.file_handler import FileInfo


class SessionState(BaseModel):
    page_title: str = "Project Assistant"
    messages: List[Message] = Field(default_factory=list)
    files_parsed: bool = False
    stop_requested: bool = False
    aggregate_info: str = ""
    train_file_path: str = ""
    test_file_path: str = ""
    file_info: FileInfo = Field(default_factory=FileInfo)
    automloutputpath: str = str(Path("autogluon_output"))
    current_model_path: str = ""
    pipeline_name: str = ""
    pipeline_state: dict = {
        "stage": "start",
        "target_column": None,
        "task_type": None,
    }

    def reset(self) -> None:
        """Reset everything"""
        self.messages = []
        self.file_info = FileInfo()
        self.aggregate_info = ""
        self.files_parsed = False
        self.stop_requested = False
        self.pipeline_name = ""
        self.automloutputpath = ""
        self.current_model_path = ""

    def add_message(self, role: str = "assistant", content: str = ""):
        self.messages.append(Message(role=role, content=content))

    def get_all_messages_by_role(
        self, roles: List[str] = ["user", "user-hidden"]
    ) -> str:
        if len(self.messages) > 0:
            return "\n".join(
                [message.content for message in self.messages if message.role in roles]
            )
        else:
            return ""

    def generate_html_from_session(self) -> BytesIO:
        rendered_messages = [
            {"role": msg.role, "content": markdown(msg.content)}
            for msg in self.messages
            if msg.role != "user-hidden" and msg.content
        ]
        html_content = render_template(
            template_name="session_html_export.html",
            page_title=self.page_title,
            messages=rendered_messages,
        )

        # Convert to BytesIO for streaming
        html_bytes: BytesIO = BytesIO(html_content.encode("utf-8"))
        html_bytes.seek(0)
        return html_bytes
