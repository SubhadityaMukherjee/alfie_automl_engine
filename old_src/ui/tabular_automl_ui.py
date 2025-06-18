from old_src.tabular_task.pipeline import BaseTabularAutoMLPipeline
from old_src.tabular_task.tasks import TabularSupervisedTimeSeriesTask

from .ui_components import SessionStateManager


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
        import os

        import streamlit as st

        try:
            SessionStateManager.append_message(
                "assistant", "🔍 Processing uploaded files..."
            )

            # Store file objects in session state before processing
            st.session_state.uploaded_file_objects = uploaded_files

            aggregate, file_paths, file_info = (
                BaseTabularAutoMLPipeline.process_uploaded_files(uploaded_files)
            )

            # Store both the processed info and original file objects
            st.session_state.aggregate_info = aggregate[:300]
            st.session_state.file_info = file_info
            st.session_state.file_paths = file_paths  # Store file paths
            st.session_state.files_parsed = True
            st.session_state.files_processed = True  # Mark as processed

            for path in file_paths:
                SessionStateManager.append_message(
                    "user-hidden",
                    f"The user uploaded a file {path} with content like {aggregate}",
                )

            if file_info["train"]:
                print(file_info["train"])
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
            # Reset processing flag if error occurs
            st.session_state.files_processed = False

    @staticmethod
    def handle_user_input(user_input, uploaded_files):
        import os

        import streamlit as st

        # First check if we need to reprocess files
        if not st.session_state.get("files_processed", False):
            if not uploaded_files:
                # Try to recover from session state if possible
                if "uploaded_file_objects" in st.session_state:
                    uploaded_files = st.session_state.uploaded_file_objects
                else:
                    SessionStateManager.append_message(
                        "assistant", "❌ No files available for processing"
                    )
                    return

            AutoMLTabularUI.handle_uploaded_files(uploaded_files)
            return

        # Target column verification
        if not st.session_state.get("target_column_confirmed", False):
            if not user_input:
                return

            # Check if we already have a proposed target column
            if "proposed_target_column" not in st.session_state:
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
                    "assistant",
                    f"🔎 Identified potential target column: {target_column}",
                )

                train_path = st.session_state.file_info.get("train")
                if not BaseTabularAutoMLPipeline.validate_column(
                    train_path, target_column
                ):
                    SessionStateManager.append_message(
                        "assistant",
                        f"⚠️ Column '{target_column}' not found in data. Please specify a valid target column.",
                    )
                    return

                # Store the proposed column and ask for confirmation
                st.session_state.proposed_target_column = target_column
                SessionStateManager.append_message(
                    "assistant",
                    f"🔍 I think the target column should be '{target_column}'. Is this correct? "
                    "(Type 'yes' to confirm, or type the correct column name if I'm wrong)",
                )
                return
            else:
                # Process user response about target column
                if user_input.lower().strip() == "yes":
                    st.session_state.target_column_confirmed = True
                    target_column = st.session_state.proposed_target_column
                    SessionStateManager.append_message(
                        "assistant", f"✅ Confirmed target column: {target_column}"
                    )
                    st.session_state.file_info["target_col"] = target_column
                else:
                    # User provided a different column name
                    new_target = user_input.strip()
                    train_path = st.session_state.file_info.get("train")
                    if BaseTabularAutoMLPipeline.validate_column(
                        train_path, new_target
                    ):
                        st.session_state.target_column_confirmed = True
                        SessionStateManager.append_message(
                            "assistant", f"✅ Using target column: {new_target}"
                        )
                        st.session_state.file_info["target_col"] = new_target
                    else:
                        SessionStateManager.append_message(
                            "assistant",
                            f"⚠️ Column '{new_target}' not found in data. Please specify a valid target column.",
                        )
                        # Clear the proposed column to restart the process
                        if "proposed_target_column" in st.session_state:
                            del st.session_state.proposed_target_column
                        return

        # After target column is confirmed, identify task type
        if not st.session_state.get("task_type_identified", False):
            AutoMLTabularUI._select_and_run_task_pipeline()
            st.session_state.task_type_identified = True
            return

    @staticmethod
    def _select_and_run_task_pipeline():
        import streamlit as st

        # Reconstruct file_info if needed
        if "file_info" not in st.session_state:
            SessionStateManager.append_message(
                "assistant", "❌ Missing file information. Please upload files again."
            )
            st.session_state.files_processed = False
            return

        task_type = BaseTabularAutoMLPipeline.infer_task_type(
            st.session_state.aggregate_info
        )

        if task_type == TabularSupervisedTimeSeriesTask:
            if not st.session_state.get("timestamp_column_confirmed", False):
                AutoMLTabularUI._handle_timestamp_column()
                return

        # Only proceed to time limit question if all previous steps are confirmed
        if not st.session_state.get("time_limit_set", False):
            time_limit = AutoMLUIWidgets.ask_time_limit()
            if time_limit is not None:
                st.session_state.time_limit_set = True
                st.session_state.time_limit = time_limit
            else:
                return

        # Verify all required information is available
        required_keys = ["file_info", "target_column_confirmed", "time_limit"]
        if any(key not in st.session_state for key in required_keys):
            SessionStateManager.append_message(
                "assistant",
                "❌ Missing required information. Please restart the process.",
            )
            return

        # Now we have everything we need to start training
        pipeline = BaseTabularAutoMLPipeline.create_pipeline(
            task_type, st.session_state.file_info
        )

        SessionStateManager.append_message(
            "assistant",
            f"⚙️ Identified Task type: {task_type}. Starting model training with time limit {st.session_state.time_limit} minutes...",
        )

        with st.spinner("Training model..."):
            if not st.session_state.get("stop_requested", False):
                try:
                    BaseTabularAutoMLPipeline.train_pipeline(
                        pipeline, st.session_state.time_limit
                    )
                except Exception as e:
                    SessionStateManager.append_message(
                        "assistant", f"❌ Error during training: {str(e)}"
                    )
                    return

        SessionStateManager.append_message(
            "assistant", "📊 Model trained successfully! Evaluating results..."
        )
        try:
            leaderboard = BaseTabularAutoMLPipeline.evaluate_pipeline(pipeline)
            if leaderboard is not None:
                SessionStateManager.append_message(
                    "assistant", "🏆 Model performance results:"
                )
                SessionStateManager.append_message(
                    "assistant", leaderboard.to_markdown()
                )
        except Exception as e:
            SessionStateManager.append_message(
                "assistant", f"❌ Error during evaluation: {str(e)}"
            )

    @staticmethod
    def _handle_timestamp_column():
        import streamlit as st

        if st.session_state.get("timestamp_column_confirmed", False):
            return

        # Check if we already have a proposed timestamp column
        if "proposed_timestamp_column" not in st.session_state:
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

            train_path = st.session_state.file_info.get("train")
            if not BaseTabularAutoMLPipeline.validate_column(train_path, timestamp_col):
                SessionStateManager.append_message(
                    "assistant",
                    f"⚠️ Column '{timestamp_col}' not found in data. Please specify a valid timestamp column.",
                )
                return

            # Store the proposed column and ask for confirmation
            st.session_state.proposed_timestamp_column = timestamp_col
            SessionStateManager.append_message(
                "assistant",
                f"🔍 I think the timestamp column should be '{timestamp_col}'. Is this correct? "
                "(Type 'yes' to confirm, or type the correct column name if I'm wrong)",
            )
            return
        else:
            # Process user response about timestamp column
            user_input = st.chat_input("Confirm timestamp column")
            if user_input:
                if user_input.lower().strip() == "yes":
                    st.session_state.timestamp_column_confirmed = True
                    timestamp_col = st.session_state.proposed_timestamp_column
                    SessionStateManager.append_message(
                        "assistant", f"✅ Confirmed timestamp column: {timestamp_col}"
                    )
                    st.session_state.file_info["time_stamp_col"] = timestamp_col
                else:
                    # User provided a different column name
                    new_timestamp = user_input.strip()
                    train_path = st.session_state.file_info.get("train")
                    if BaseTabularAutoMLPipeline.validate_column(
                        train_path, new_timestamp
                    ):
                        st.session_state.timestamp_column_confirmed = True
                        SessionStateManager.append_message(
                            "assistant", f"✅ Using timestamp column: {new_timestamp}"
                        )
                        st.session_state.file_info["time_stamp_col"] = new_timestamp
                    else:
                        SessionStateManager.append_message(
                            "assistant",
                            f"⚠️ Column '{new_timestamp}' not found in data. Please specify a valid timestamp column.",
                        )
                        # Clear the proposed column to restart the process
                        if "proposed_timestamp_column" in st.session_state:
                            del st.session_state.proposed_timestamp_column
                        return
