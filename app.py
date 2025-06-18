import nest_asyncio
import streamlit as st

nest_asyncio.apply()


from src.session_state_handler import SessionState
from src.streamlit_ui_handler import build_ui_with_chat


def main():
    # Persist session state across reruns (Streamlit specific)
    if "session" not in st.session_state:
        st.session_state.session = SessionState()

    session_state = st.session_state.session
    ui_builder = build_ui_with_chat(session_state=session_state)
    ui_builder.build_ui()


if __name__ == "__main__":
    main()
