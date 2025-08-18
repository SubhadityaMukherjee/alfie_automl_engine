from abc import ABC, abstractmethod
from typing import List

import nest_asyncio

nest_asyncio.apply()


class BaseUITemplate(ABC):
    def __init__(self, session_state) -> None:
        self.session_state = session_state

    @abstractmethod
    def set_page_config(self, title: str, layout: str) -> None:
        pass

    @abstractmethod
    def show_title(self, title: str) -> None:
        pass

    @abstractmethod
    def show_subheader(self, text: str) -> None:
        pass

    @abstractmethod
    def sidebar_header(self, text: str) -> None:
        pass

    @abstractmethod
    def sidebar_button(self, label: str) -> bool:
        pass

    @abstractmethod
    def download_button(self, label: str, data, file_name: str, mime: str) -> None:
        pass

    @abstractmethod
    def chat_input(self, placeholder: str) -> str:
        pass

    @abstractmethod
    def text_input(self, label: str, key: str) -> str:
        pass

    @abstractmethod
    def selectbox(
        self, label: str, options: List[str], key: str, index: int = 0
    ) -> str:
        pass

    @abstractmethod
    def file_uploader(self, label: str, accept_multiple_files: bool, key: str):
        pass

    @abstractmethod
    def container(self):
        pass

    @abstractmethod
    def empty(self):
        pass

    @abstractmethod
    def chat_message(self, role: str, content: str) -> None:
        pass

    @abstractmethod
    def markdown(self, text: str) -> None:
        pass

    @abstractmethod
    def form(self, text: str) -> None:
        pass

    @abstractmethod
    def form_submit_button(self, text: str) -> None:
        pass

    @abstractmethod
    def spinner(self, text: str):
        pass

    @abstractmethod
    def warning(self, text: str) -> None:
        pass

    @abstractmethod
    def rerun(self) -> None:
        pass
