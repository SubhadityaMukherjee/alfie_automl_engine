import uuid

import requests
import streamlit as st
from automl_engine.models import SessionState
from automl_engine.pipelines import PIPELINES
from ui_2.streamlit_handler import StreamlitUI

BACKEND = "http://localhost:8000"
session_id = st.session_state.get("session_id") or str(uuid.uuid4())
st.session_state["session_id"] = session_id


def get_session():
    res = requests.get(f"{BACKEND}/session/{session_id}")
    return res.json()


def send_chat(prompt: str, uploaded_files, context: str, handle_intent: bool):
    if uploaded_files is None:
        files = []
    if len(uploaded_files) > 0:
        files = [("files", (f.name, f, f.type)) for f in uploaded_files]
    else:
        files = []
    data = {"session_id": session_id, "prompt": prompt, "context": context}
    if handle_intent:
        res = requests.post(f"{BACKEND}/chat/intent_recog/", data=data, files=files)
    else:
        res = requests.post(f"{BACKEND}/chat/", data=data, files=files)
    return res.json()


def reset_chat():
    # st.session_state["session_id"] = str(uuid.uuid4())
    ...


def build_ui():
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
    session_state = st.session_state.session

    if "stage" not in st.session_state:
        st.session_state.stage = "intent recognition"

    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""

    ui = StreamlitUI(session_state=session_state)

    ui.show_title("ALFIE AutoML Engine")
    ui.show_subheader(
        "Note that the AI can often make mistakes. Before doing anything important, please verify it.\nEg tasks include: tabular classification, website accessibility"
    )
    ui.sidebar_header("Extras")

    if st.sidebar.button("🧹 Reset Session"):
        reset_chat()
        st.session_state.stage = "intent recognition"
        st.session_state.user_prompt = ""
        return

    chat_area = ui.container()
    user_input = ui.chat_input("What do you want to do? Eg: tabular classification...")

    if user_input:
        st.session_state.user_prompt = user_input
        st.session_state.stage = "intent recognition"

    with chat_area:
        output_placeholder = ui.container()

    pipeline_with_prompt = "\n".join(
        f"pipeline: {pipeline}, possible_query: {PIPELINES.get(pipeline, '').__doc__}"
        for pipeline in PIPELINES
    )

    if st.session_state.stage == "intent recognition":
        user_prompt = st.session_state.user_prompt
        if user_prompt:
            with st.spinner("Thinking..."):
                intent_recog = send_chat(
                    f"user query: {user_prompt}",
                    uploaded_files=[],
                    context=pipeline_with_prompt,
                    handle_intent=True,
                )
                if intent_recog.get("verified"):
                    st.session_state.intent_reply = intent_recog.get("reply", "")
                    st.session_state.stage = "get_required"

    if st.session_state.stage == "get_required":
        form_components_to_show = requests.post(
            f"{BACKEND}/get_pipeline_requirements/{st.session_state.intent_reply}",
        ).json()
        reply = form_components_to_show[
            "reply"
        ]  # this gets all the required components needed to proceed further
        session_form = {}

        if reply is not None:
            with output_placeholder:
                for component, type_of_component in reply.items():
                    if type_of_component == "file_upload_multi":
                        uploaded_files = ui.file_uploader(
                            label=component,
                            key=component,
                            accept_multiple_files=True,
                        )
                        if uploaded_files:
                            session_form[component] = uploaded_files

                    elif type_of_component == "file_upload_train":
                        train_csv = ui.file_uploader(
                            label=component,
                            key=component,
                            accept_multiple_files=False,
                        )
                        if train_csv:
                            session_form[component] = train_csv

                    elif type_of_component == "file_upload_test":
                        test_csv = ui.file_uploader(
                            label=component,
                            key=component,
                            accept_multiple_files=False,
                        )
                        if test_csv:
                            session_form[component] = test_csv

                    elif component == "target_col":
                        target_col = ui.text_input(label=component, key=component)
                        if target_col:
                            session_form[component] = target_col

                    elif component == "timestamp_col":
                        timestamp_col = ui.text_input(label=component, key=component)
                        if timestamp_col:
                            session_form[component] = timestamp_col

        if session_form:
            session_state.session_form = session_form
            st.session_state.stage = "pipeline"

    if st.session_state.stage == "pipeline":
        # TODO: handle pipeline execution
        pass

    # Render session messages
    session = get_session()
    for msg in session["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


if __name__ == "__main__":
    build_ui()
