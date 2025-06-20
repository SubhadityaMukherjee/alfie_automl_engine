import os
import io
import tempfile
from pathlib import Path
from session_state_handler import SessionState, Message
from datetime import datetime
from file_handler import FileInfo
import pytest

def test_session_state_with_values():
    message = Message(role="user", content="Hello", timestamp=datetime.utcnow())
    # TODO file_info test one doesnt work
    # file_info = FileInfo()
    state = SessionState(
        page_title="Test Page",
        messages=[message],
        files_parsed=True,
        stop_requested=True,
        aggregate_info="Summary",
        # file_info=file_info,
        automloutputpath="/tmp/test_output",
        pipeline_name="AutoML",
    )

    assert state.page_title == "Test Page"
    assert state.messages[0].content == "Hello"
    assert state.files_parsed is True
    assert state.stop_requested is True
    assert state.aggregate_info == "Summary"
    assert state.automloutputpath == "/tmp/test_output"
    assert state.pipeline_name == "AutoML"


@pytest.fixture
def mock_session_state():
    state = SessionState()
    state.add_message(role="user", content="Hello")
    state.add_message(role="user", content="Hello2")
    return state


def test_add_message(mock_session_state):
    assert mock_session_state.messages[-1].content == "Hello2"


def test_get_all_messages_by_role(mock_session_state):
    assert mock_session_state.get_all_messages_by_role() == "Hello\nHello2"

def test_generate_html_from_session(mock_session_state):
    assert isinstance(mock_session_state.generate_html_from_session(), io.BytesIO)