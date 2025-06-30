import streamlit as st
import requests
import uuid
from ui.streamlit_handler import StreamlitUI
from automl_engine.models import SessionState
from automl_engine.pipelines import PIPELINES

BACKEND = "http://localhost:8000"
session_id = st.session_state.get("session_id") or str(uuid.uuid4())
st.session_state["session_id"] = session_id

def get_session():
    res = requests.get(f"{BACKEND}/session/{session_id}")
    return res.json()

def send_chat(prompt:str, uploaded_files, context:str, handle_intent: bool):
    if uploaded_files is None:
        files =[]
    if len(uploaded_files)>0:
        files = [("files", (f.name, f, f.type)) for f in uploaded_files] 
    else: files = []
    data = {"session_id": session_id, "prompt": prompt, "context":context}
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

    ui = StreamlitUI(session_state=session_state)

    ui.show_title("ALFIE AutoML Engine")
    ui.show_subheader(
        "Note that the AI can often make mistakes. Before doing anything important, please verify it.\nEg tasks include: tabular classification, website accessibility"
    )
    ui.sidebar_header("Extras")

    if st.sidebar.button("🧹 Reset Session"):
        reset_chat()

    # uploaded_files = st.file_uploader("📂 Upload files", accept_multiple_files=True)
    user_prompt = st.chat_input("What do you want to do? Eg: tabular classification...")
    pipeline_with_prompt = "\n".join(f"pipeline: {pipeline}, possible_query: {PIPELINES.get(pipeline, '').__doc__}" for pipeline in PIPELINES)

    stage = "intent recognition"
    if user_prompt:
        with st.spinner("Thinking..."):
            intent_recog = send_chat(f"user query: {user_prompt}", uploaded_files=[], context = pipeline_with_prompt, handle_intent = True)
            if intent_recog.get('verified'):
                stage = "pipeline"
            
        # if stage == "pipeline":


    # After sending (or on first load), fetch updated session
    session = get_session()

    # Render all messages from session only (no local additions)
    for msg in session["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if __name__ == "__main__":
    build_ui()
