from typing import List

import nest_asyncio
from markdown import markdown

nest_asyncio.apply()
from io import BytesIO

from pydantic import BaseModel, Field

from src.chat_handler import Message
from src.file_handler import FileInfo


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
    
    def get_all_messages_by_role(self, roles:List[str] = ["user", "user-hidden"]) -> str:
        if len(self.messages) > 0:
            return "\n".join([message.content for message in self.messages if message.role in roles])
        else:
            return ""

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
