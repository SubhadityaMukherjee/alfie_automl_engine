from typing import Any, Dict, List, Optional, Union

import nest_asyncio
import streamlit as st
from src.llm_task.wcag import WCAGPipeline

from src.ui.ui_components import SessionStateManager

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

        result = WCAGPipeline.handle_user_input(user_input, uploaded_files)

        if result["status"] != "success":
            SessionStateManager.append_message("assistant", f"❌ {result['message']}")
            return

        for file_result in result["results"]:
            filename = file_result["filename"]
            num_chunks = file_result["num_chunks"]
            SessionStateManager.append_message(
                "assistant", f"📂 Processing `{filename}` in {num_chunks} chunk(s)..."
            )

            for chunk_data in file_result["feedback"]:
                if st.session_state.get("stop_requested", False):
                    SessionStateManager.append_message(
                        "assistant", "🛑 Processing stopped by user."
                    )
                    return

                with st.spinner(
                    f"🔎 Analyzing `{filename}` - chunk {chunk_data['chunk_index'] + 1}/{num_chunks}..."
                ):
                    feedback = f"""
📄 **Chunk {chunk_data['chunk_index']+1}/{num_chunks} of `{filename}` (lines {chunk_data['start_line']}-{chunk_data['end_line']})**  
{chunk_data['response']}
"""
                    SessionStateManager.append_message("assistant", feedback)

            avg_score = (
                file_result["average_score"]
                if file_result["average_score"] is not None
                else "N/A"
            )
            summary = f"✅ Finished analyzing `{filename}`.\n\n**Average WCAG Score:** {avg_score}/10"
            SessionStateManager.append_message("assistant", summary)
