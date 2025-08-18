from typing import List

import nest_asyncio
import streamlit as st

nest_asyncio.apply()

from contextlib import contextmanager

from streamlit.commands.page_config import Layout
from ui_2.ui_template import BaseUITemplate


class StreamlitUI(BaseUITemplate):
    def set_page_config(self, title: str, layout: Layout) -> None:
        st.set_page_config(page_title=title, layout=layout)

    def show_title(self, title: str) -> None:
        st.title(title)

    def show_subheader(self, text: str) -> None:
        st.subheader(text)

    def sidebar_header(self, text: str) -> None:
        st.sidebar.header(text)

    def sidebar_button(self, label: str) -> bool:
        return st.sidebar.button(label)

    def download_button(self, label: str, data, file_name: str, mime: str) -> None:
        st.download_button(label=label, data=data, file_name=file_name, mime=mime)

    def chat_input(self, placeholder: str) -> str | None:
        return st.chat_input(placeholder)

    def form(self, key: str):
        return st.form(key=key)

    def form_submit_button(self, label: str):
        return st.form_submit_button(label=label)

    def selectbox(
        self, label: str, options: List[str], key: str, index: int = 0
    ) -> str:
        return st.selectbox(label, options, key=key, index=index)

    def file_uploader(self, label: str, accept_multiple_files: bool, key: str):
        return st.file_uploader(
            label, accept_multiple_files=accept_multiple_files, key=key
        )

    def container(self):
        return st.container()

    def empty(self):
        return st.empty()

    @contextmanager
    def chat_message(self, role: str, content: str = ""):
        with st.chat_message(role):
            yield

    def markdown(self, text: str) -> None:
        st.markdown(text)

    @contextmanager
    def spinner(self, text: str):
        with st.spinner(text):
            yield

    def warning(self, text: str) -> None:
        st.warning(text)

    def rerun(self) -> None:
        st.rerun()

    def text_input(self, label: str, key: str) -> str:
        return st.text_input(label=label, key=key)
