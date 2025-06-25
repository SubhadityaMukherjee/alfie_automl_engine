from typing import Any, Dict, List, Optional, Union

import nest_asyncio
import streamlit as st

from old_src.file_processing.reader import FileHandler
from old_src.llm_task.wcag import WCAGPipeline
from old_src.ui.ui_components import SessionStateManager

nest_asyncio.apply()


class WCAGPipelineUI:

    @staticmethod
    def process_uploaded_files(uploaded_files):
        result = WCAGPipeline.process_uploaded_files(uploaded_files)
        if result["status"] == "success":
            for path in result["file_paths"]:
                SessionStateManager.append_message(
                    "user-hidden", f"The user uploaded a file {path}"
                )
            return result["file_paths"]
        else:
            SessionStateManager.append_message("assistant", f"❌ {result['message']}")
            return None

    @staticmethod
    def handle_user_input(user_input: str, uploaded_files) -> Dict[str, Any] | None:
        SessionStateManager.append_message(
            "assistant", "🔍 Processing uploaded files..."
        )

        if st.session_state.get("stop_requested", False):
            SessionStateManager.append_message(
                "assistant", "🛑 Processing stopped by user."
            )
            return

        try:
            files = FileHandler.read_each_file(uploaded_files)
        except Exception as e:
            SessionStateManager.append_message(
                "assistant", f"❌ Error reading files: {e}"
            )
            return

        for filename, content in files.items():
            SessionStateManager.append_message(
                "assistant", f"📂 Processing `{filename}`..."
            )

            for chunk_result in WCAGPipeline.process_file(filename, content):
                if st.session_state.get("stop_requested", False):
                    SessionStateManager.append_message(
                        "assistant", "🛑 Processing stopped by user."
                    )
                    return

                if "summary" in chunk_result:
                    summary = f"✅ Finished analyzing `{chunk_result['filename']}`.\n\n**Average WCAG Score:** {chunk_result['average_score']}/10"
                    SessionStateManager.append_message("assistant", summary)
                else:
                    with st.spinner(
                        f"🔎 Analyzing `{filename}` - chunk {chunk_result['chunk_index'] + 1}/{chunk_result['num_chunks']}..."
                    ):
                        feedback = f"""
    📄 **Chunk {chunk_result['chunk_index']+1}/{chunk_result['num_chunks']} of `{filename}` (lines {chunk_result['start_line']}-{chunk_result['end_line']})**  
    {chunk_result['response']}
    """
                        SessionStateManager.append_message("assistant", feedback)
