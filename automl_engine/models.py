from typing import ClassVar, List, Union, Dict
from pydantic import BaseModel, Field
from io import BytesIO
from pathlib import Path
from automl_engine.utils import render_template
from markdown import markdown
from datetime import datetime
from typing import Optional
from jinja2 import Environment, FileSystemLoader

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

class FileInfo(BaseModel):
    train_file: Path = Path()
    test_file: Path = Path()
    target_col: str = ""
    time_stamp_col: str = ""

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
    jinja_environment : ClassVar[Environment] = Environment(loader=FileSystemLoader(Path("automl_engine/prompt_templates/")))

    def reset(self) -> None:
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

    def get_all_messages_by_role(self, roles: List[str]) -> str:
        return "\n".join(
            [m.content for m in self.messages if m.role in roles]
        )

    def generate_html_from_session(self) -> BytesIO:
        rendered_messages = [
            {"role": msg.role, "content": markdown(msg.content)}
            for msg in self.messages if msg.role != "user-hidden" and msg.content
        ]
        html_content = render_template(
            jinja_environment=SessionState.jinja_environment,
            template_name="session_html_export.html", page_title=self.page_title, messages=rendered_messages
        )
        return BytesIO(html_content.encode("utf-8"))

