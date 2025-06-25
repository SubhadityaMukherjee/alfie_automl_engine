import shutil
from typing import List, Union

import nest_asyncio

nest_asyncio.apply()

from src.chat_handler import ChatHandler, Message
from src.pipelines.base import PipelineRegistry
from src.ui.streamlit_handler import StreamlitUI


class build_ui_with_chat:
    def __init__(self, session_state) -> None:
        self.session_state = session_state
        self.ui = StreamlitUI(session_state=self.session_state)
        self.PIPELINES = {"-- Select a Pipeline --": None, **PipelineRegistry.get_all()}

    def build_ui(self) -> None:
        self.ui.set_page_config(self.session_state.page_title, layout="wide")
        self.ui.show_title("ALFIE AutoML Engine")
        self.ui.show_subheader(
            "Note that the AI can often make mistakes. Before doing anything important, please verify it."
        )
        self.ui.sidebar_header("Extras")

        chat_area = self.ui.container()
        prompt = self.ui.chat_input(
            "What would you like help with? Upload your files for context"
        )

        with chat_area:
            self.output_placeholder = self.ui.empty()
            self.display_messages()
            pipeline_name = self.ui.selectbox(
                "Choose a Pipeline",
                list(self.PIPELINES.keys()),
                key="pipeline_selector",
                index=0,
            )
            uploaded_files = self.ui.file_uploader(
                "📂 Upload files", accept_multiple_files=True, key="file_uploader"
            )
            handler_class = self.PIPELINES.get(pipeline_name)

        if prompt:
            self.session_state.add_message(role="user", content=prompt)
            intent = self.classify_intent_with_chat(prompt)

            if intent == "general":
                response = ChatHandler.chat(prompt)
                self.session_state.add_message(role="assistant", content=response)

            if intent == "pipeline" and handler_class:
                if pipeline_name != "-- Select a Pipeline --":
                    chosen_pipeline_class = handler_class(
                        session_state=self.session_state,
                        output_placeholder_ui_element=self.output_placeholder,
                    )
                    if not any(
                        m.content == chosen_pipeline_class.initial_display_message
                        for m in self.session_state.messages
                    ):
                        self.session_state.add_message(
                            role="assistant",
                            content=chosen_pipeline_class.initial_display_message,
                        )
                    with self.ui.spinner("Processing..."):
                        chosen_pipeline_class.main_flow(prompt, uploaded_files)

            else:
                self.session_state.add_message(
                    role="assistant",
                    content="❓ I wasn't sure how to route this. Please try rephrasing or choose a pipeline.",
                )

            self.ui.rerun()

        self.generate_sidebar()

    def classify_intent_with_chat(self, prompt: str) -> str:
        query = (
            f"The user said: '{prompt}'. "
            "Should this be handled as a general question (like 'what is AutoML?' or 'how do I train a model?') "
            "or as a pipeline-related task (e.g., involving dataset processing, file upload, or model training)? "
            "Pipeline related could also be something like classify X column, -> pipeline"
            "Regression on X-> pipeline"
            "General question could be : How do I do X? How do I modify X to do Y"
            "How do I do X better -> general"
            "Check -> pipeline"
            "Check guidelines -> pipeline"
            "Respond with exactly one word: 'general' or 'pipeline'."
        )
        return ChatHandler.chat(query).strip().lower()

    def generate_sidebar(self) -> None:
        if self.ui.sidebar_button("🧹clear"):
            self.session_state.reset()
            self.ui.rerun()

        if self.ui.sidebar_button("🛑 stop"):
            self.session_state.stop_requested = True
            self.ui.warning("Stop requested. Trying to halt processing...")
            self.ui.rerun()

        if self.ui.sidebar_button("📄 download chat"):
            html_chat = self.session_state.generate_html_from_session()
            try:
                self.ui.download_button(
                    label="📄 Download Conversation as html",
                    data=html_chat,
                    file_name="chat_session.html",
                    mime="text/html",
                )
            except Exception as e:
                self.ui.warning(str(e))

        # TODO Only enable this for automl task
        if self.ui.sidebar_button("📄 download models after leaderboard"):
            zip_filename = "best_model.zip"
            shutil.make_archive(
                "best_model", "zip", self.session_state.automloutputpath
            )

            # Streamlit download button
            with open(zip_filename, "rb") as f:
                self.ui.download_button(
                    label="📥 Download Best Model",
                    data=f,
                    file_name=zip_filename,
                    mime="application/zip",
                )

    def append_message(self, role: str, content: str, display: bool = True) -> None:
        message = Message(role=role, content=content)
        self.session_state.messages.append(message)

    def get_conversation_text_by_role(
        self, role: Union[str, List[str]] = ["user", "user-hidden"]
    ) -> str:
        if isinstance(role, str):
            role = [role]
        return "\n".join(
            msg.content for msg in self.session_state.messages if msg.role in role
        )

    def display_messages(self) -> None:
        for msg in self.session_state.messages:
            if msg.role != "user-hidden":
                with self.ui.chat_message(msg.role):
                    self.ui.markdown(msg.content)
