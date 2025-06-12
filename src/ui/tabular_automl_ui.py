from .ui_components import SessionStateManager
from src.tabular_task.pipeline import BaseTabularAutoMLPipeline
from src.tabular_task.tasks import TabularSupervisedTimeSeriesTask

class AutoMLUIWidgets:
    @staticmethod
    def ask_time_limit():
        import streamlit as st

        selection = st.selectbox(
            "How fast do you want a model? (Longer might be better)",
            ["20 seconds", "5 min", "10 min", "1 hour"],
        )
        return {
            "20 seconds": 20,
            "5 min": 5 * 60,
            "10 min": 10 * 60,
            "1 hour": 60 * 60,
        }.get(selection)


class AutoMLTabularUI:
    @staticmethod
    def handle_uploaded_files(uploaded_files):
        import streamlit as st

        try:
            SessionStateManager.append_message(
                "assistant", "🔍 Processing uploaded files..."
            )
            aggregate, file_paths, file_info = (
                BaseTabularAutoMLPipeline.process_uploaded_files(uploaded_files)
            )

            st.session_state.aggregate_info = aggregate[:300]
            st.session_state.file_info = file_info
            st.session_state.files_parsed = True

            for path in file_paths:
                SessionStateManager.append_message(
                    "user-hidden",
                    f"The user uploaded a file {path} with content like {aggregate}",
                )

            if file_info["train"]:
                SessionStateManager.append_message(
                    "assistant", "✅ Found training data file"
                )
            else:
                SessionStateManager.append_message(
                    "assistant", "⚠️ No train.csv file provided"
                )

            if file_info["test"]:
                SessionStateManager.append_message(
                    "assistant", "✅ Found test data file"
                )
            else:
                SessionStateManager.append_message(
                    "assistant", "ℹ️ No test file provided"
                )

            SessionStateManager.append_message(
                "assistant",
                "📊 Files processed successfully. Please tell me about your target column.",
            )

        except Exception as e:
            SessionStateManager.append_message(
                "assistant", f"❌ Error processing files: {str(e)}"
            )

    @staticmethod
    def handle_user_input(user_input, uploaded_files):
        import streamlit as st

        if not uploaded_files:
            return

        AutoMLTabularUI.handle_uploaded_files(uploaded_files)

        SessionStateManager.append_message(
            "assistant", "🤔 Analyzing your input for target column..."
        )
        conversation_text = SessionStateManager.get_conversation_text()
        target_column = BaseTabularAutoMLPipeline.detect_target_column(
            conversation_text
        )

        if target_column.lower() == "no":
            SessionStateManager.append_message(
                "assistant",
                "❓ I couldn't identify the target column. Please specify which column we should predict.",
            )
            return

        SessionStateManager.append_message(
            "assistant", f"🔎 Identified potential target column: {target_column}"
        )
        SessionStateManager.append_message("assistant", "⚙️ Validating target column...")

        train_path = st.session_state.file_info.get("train")
        if not BaseTabularAutoMLPipeline.validate_column(train_path, target_column):
            SessionStateManager.append_message(
                "assistant",
                f"⚠️ Column '{target_column}' not found in data. Please specify a valid target column.",
            )
            return

        SessionStateManager.append_message(
            "assistant", f"✅ Target column '{target_column}' validated successfully!"
        )
        st.session_state.file_info["target_col"] = target_column

        AutoMLTabularUI._select_and_run_task_pipeline()

    @staticmethod
    def _select_and_run_task_pipeline():
        import streamlit as st

        task_type = BaseTabularAutoMLPipeline.infer_task_type(
            st.session_state.aggregate_info
        )
        SessionStateManager.append_message(
            "assistant", f"🔧 Creating {task_type} pipeline..."
        )

        if task_type == TabularSupervisedTimeSeriesTask:
            AutoMLTabularUI._handle_timestamp_column()

        pipeline = BaseTabularAutoMLPipeline.create_pipeline(
            task_type, st.session_state.file_info
        )
        time_limit = AutoMLUIWidgets.ask_time_limit()

        if st.session_state.get("stop_requested", False):
            SessionStateManager.append_message(
                "assistant", "🛑 Processing stopped by user."
            )
            return

        SessionStateManager.append_message(
            "assistant",
            f"⚙️ Identified Task type: {task_type}. Training model... (this may take a few minutes)",
        )

        with st.spinner("Training model..."):
            if not st.session_state.get("stop_requested", False):
                BaseTabularAutoMLPipeline.train_pipeline(pipeline, time_limit)

        SessionStateManager.append_message(
            "assistant", "📊 Model trained successfully! Evaluating results..."
        )
        leaderboard = BaseTabularAutoMLPipeline.evaluate_pipeline(pipeline)
        if leaderboard is not None:
            SessionStateManager.append_message(
                "assistant", "🏆 Model performance results:"
            )
            SessionStateManager.append_message("assistant", leaderboard.to_markdown())

    @staticmethod
    def _handle_timestamp_column():
        import streamlit as st

        SessionStateManager.append_message(
            "assistant", "📅 Trying to identify timestamp column..."
        )
        conversation_text = SessionStateManager.get_conversation_text()
        timestamp_col = BaseTabularAutoMLPipeline.detect_timestamp_column(
            conversation_text
        )

        if timestamp_col.lower() == "no":
            SessionStateManager.append_message(
                "assistant",
                "❓ I couldn't identify the time stamp column. Please specify which column we should use.",
            )
            return

        SessionStateManager.append_message(
            "assistant", f"🔎 Identified potential timestamp column: {timestamp_col}"
        )
        train_path = st.session_state.file_info.get("train")
        if not BaseTabularAutoMLPipeline.validate_column(train_path, timestamp_col):
            SessionStateManager.append_message(
                "assistant",
                f"⚠️ Column '{timestamp_col}' not found in data. Please specify a valid timestamp column.",
            )
            return

        st.session_state.file_info["time_stamp_col"] = timestamp_col
        SessionStateManager.append_message(
            "assistant",
            f"✅ Timestamp column '{timestamp_col}' validated successfully!",
        )

