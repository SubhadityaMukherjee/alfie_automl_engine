from abc import ABC
from typing import Any, Dict


class BasePipeline(ABC):
    
    def __init__(self, session_state, output_placeholder_ui_element) -> None:
        super().__init__()
        self.session_state = session_state
        self.output_placeholder_ui_element = output_placeholder_ui_element

    def main_flow(self, user_input: str, uploaded_files) -> Dict[str, Any] | None: ...

    @staticmethod
    def return_basic_prompt() -> str: ...
