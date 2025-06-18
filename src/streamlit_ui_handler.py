from typing import List

import nest_asyncio
import streamlit as st

nest_asyncio.apply()


from src.chat_handler import Message
from src.pipelines import AutoMLTabularPipeline, WCAGPipeline
from src.session_state_handler import SessionState


class build_ui_with_chat:
    def __init__(self, session_state: SessionState) -> None:
        self.session_state: SessionState = session_state
        self.PIPELINES = {
            "-- Select a Pipeline --": None,
            "WCAG Guidelines": WCAGPipeline,
            "AutoML Tabular": AutoMLTabularPipeline,
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
