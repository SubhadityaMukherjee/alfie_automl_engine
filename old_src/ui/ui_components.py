import streamlit as st


class SessionStateManager:
    @staticmethod
    def initialize_state():
        st.set_page_config(page_title="Project Assistant", layout="wide")
        st.title("ALFIE AutoML Engine")
        st.subheader(
            "Note that the AI can often make mistakes. Before doing anything important, please verify it."
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "file_info" not in st.session_state:
            st.session_state.file_info = {
                "train": "",
                "test": "",
                "target_col": "",
                "time_stamp_col": "",
            }
        if "files_parsed" not in st.session_state:
            st.session_state.files_parsed = False
        if "stop_requested" not in st.session_state:
            st.session_state.stop_requested = False
        if "aggregate_info" not in st.session_state:
            st.session_state.aggregate_info = ""

    @staticmethod
    def append_message(role: str, content: str, display: bool = True):
        """Append message to session state and optionally display it"""
        st.session_state.messages.append({"role": role, "content": content})
        if display and role != "user-hidden":
            with st.chat_message(role):
                st.markdown(content)

    @staticmethod
    def clear_conversation():
        st.session_state.clear()
        st.rerun()

    @staticmethod
    def get_conversation_text() -> str:
        """Get all user and user-hidden messages as text"""
        return "\n".join(
            msg["content"]
            for msg in st.session_state.messages
            if msg["role"] in ["user", "user-hidden"]
        )


class GeneralUIComponents:
    @staticmethod
    def display_sidebar():
        st.sidebar.header("Conversation History")
        if "messages" in st.session_state:
            for msg in st.session_state.messages:
                if msg["role"] in ["user", "assistant"]:  # Skip hidden messages
                    role = "👤" if msg["role"] == "user" else "🤖"
                    content = (
                        msg["content"]
                        if isinstance(msg["content"], str)
                        else str(msg["content"])
                    )
                    st.sidebar.caption(
                        f"{role}: {content[:50]}{'...' if len(content) > 50 else ''}"
                    )
        st.sidebar.divider()

        if st.sidebar.button("Clear Conversation"):
            SessionStateManager.clear_conversation()

    @staticmethod
    def render_chat():
        """Render only new chat messages"""
        # if "messages" in st.session_state:
        # Only render messages that haven't been displayed yet
        for msg in st.session_state.messages:
            if msg["role"] != "user-hidden" and not msg.get("displayed", False):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                msg["displayed"] = True  # Mark as displayed
