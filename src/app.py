# app.py
import streamlit as st
import asyncio
from agent import InteractiveProjectAgent
from pathlib import Path
import tempfile
import mimetypes

st.set_page_config(page_title="Project Assistant", layout="centered")
st.title("🤖 Interactive Project Assistant")

agent = InteractiveProjectAgent()


def read_file_content(file_path: Path, mime: str) -> str:
    try:
        if mime.startswith("text/") or file_path.suffix.lower() in [
            ".html",
            ".css",
            ".py",
            ".json",
            ".csv",
        ]:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        elif file_path.suffix.lower() == ".docx":
            return agent.read_word_document(file_path)
        else:
            return f"📦 Binary or unsupported file type: {file_path.name} ({mime})"
    except Exception as e:
        return f"⚠️ Failed to read {file_path.name}: {e}"


with st.form("user_input_form"):
    user_query = st.text_area("💬 What would you like help with?", height=150)
    uploaded_files = st.file_uploader(
        "📂 Upload any number of files", type=None, accept_multiple_files=True
    )
    submitted = st.form_submit_button("Submit")

if submitted and user_query:
    aggregated_context = ""

    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.name).suffix
            ) as tmp:
                print(f"Processing file: {file.name}")
                tmp.write(file.read())
                tmp_path = Path(tmp.name)
                mime_type, _ = mimetypes.guess_type(tmp_path.name)
                mime_type = mime_type or "application/octet-stream"

                content = read_file_content(tmp_path, mime_type)
                aggregated_context += f"\n---\nAgent has been given a file of type {mime_type} and the content is as follows: {file.name} ({mime_type})\n{content}\n"
    # print(f"Aggregated context: {aggregated_context}")

    with st.spinner("Thinking... 🤔"):
        result_text = asyncio.run(
            agent.ask_agent(user_query, context=aggregated_context)
        )
        parsed = agent.parse_intent_from_text(result_text)
    if st.sidebar.button("🔁 Reset Conversation"):
        agent.conversation_history.clear()
        st.success("Conversation memory reset.")
    st.subheader("🧠 Assistant Response")
    st.markdown(parsed["final_answer"])

    st.markdown("---")
    st.subheader("🧾 Parsed Understanding")
    st.write(
        {
            "Has enough info?": parsed["enough_information"],
            "Needs files or data?": parsed["needs_files"],
            "Wants to train model?": parsed["wants_to_train"],
            "Use existing model?": parsed["use_existing_model"],
            "Extra question": parsed["extra_question"],
        }
    )
    st.markdown("---")
    st.subheader("🗂️ Chat History")
    st.markdown(agent.get_chat_history())

    # st.write(result_text)

    if parsed["extra_question"]:
        st.info(f"📝 The assistant needs more info: {parsed['extra_question']}")
