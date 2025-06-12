from typing import Any, Dict, List, Optional, Union

import nest_asyncio
import streamlit as st

from src.ui.ui_components import SessionStateManager, GeneralUIComponents
from src.ui.tabular_automl_ui import AutoMLTabularUI
from src.ui.wcag_pipeline_ui import WCAGPipelineUI

nest_asyncio.apply()




def main():
    SessionStateManager.initialize_state()
    GeneralUIComponents.display_sidebar()

    col1, col2 = st.columns([3, 1])

    # Map pipeline names to their corresponding UI handler classes
    PIPELINES = {
        "AutoML Tabular": AutoMLTabularUI,
        "WCAG Guidelines": WCAGPipelineUI,
    }

    def render_pipeline_ui(pipeline_label, handler_class):
        uploaded_files = st.file_uploader(
            "📂 Upload files", accept_multiple_files=True, key="file_uploader"
        )

        prompt = st.chat_input("What would you like help with? Upload your files for context")
        if prompt:
            SessionStateManager.append_message("user", prompt)
            handler_class.handle_user_input(prompt, uploaded_files)
            if hasattr(st.session_state, "stop_requested") and st.session_state.stop_requested:
                st.session_state.stop_requested = False
            st.rerun()

    with col1:
        GeneralUIComponents.render_chat()

        pipeline_name = st.selectbox(
            "Choose a Pipeline",
            list(PIPELINES.keys()),
            key="pipeline_selector",
        )

        handler_class = PIPELINES.get(pipeline_name)
        if handler_class:
            render_pipeline_ui(pipeline_name, handler_class)

    with col2:
        if st.button("🛑 Stop Processing"):
            st.session_state.stop_requested = True
            st.warning("Stop requested. Trying to halt processing...")



if __name__ == "__main__":
    main()
