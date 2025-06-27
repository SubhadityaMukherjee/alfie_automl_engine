from pathlib import Path
import nest_asyncio
import streamlit as st

nest_asyncio.apply()


from src.session_state_handler import SessionState
from src.ui.ui_handler import build_ui_with_chat

from redislite import Redis
import uuid

# >>> redis_connection = Redis('/tmp/redis.db')
# >>> redis_connection.keys()
# []
# >>> redis_connection.set('key', 'value')
# True
# >>> redis_connection.get('key')
# 'value'


def generate_redis_path() -> str:
    return str(Path(f"/tmp/{str(uuid.uuid1())}/redis.db"))


def main():
    # Persist session state across reruns (Streamlit specific)
    if "session" not in st.session_state:
        st.session_state.session = SessionState()

    session_state = st.session_state.session
    ui_builder = build_ui_with_chat(session_state=session_state)
    ui_builder.build_ui()


if __name__ == "__main__":
    main()
