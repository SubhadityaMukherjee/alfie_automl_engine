from unittest.mock import patch
from datetime import datetime
from chat_handler import ChatHandler, Message  
import time

# Test: successful chat response
@patch("ollama.chat")
def test_chat_success(mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": "Hello there!"
        }
    }
    response = ChatHandler.chat("Hi!")
    assert response == "Hello there!"
    mock_chat.assert_called_once()

# Test: model not found error handling
@patch("ollama.chat")
def test_chat_model_not_found(mock_chat):
    mock_chat.side_effect = Exception("model 'gemma3:4b' not found")
    response = ChatHandler.chat("Hi!")
    assert "Model 'gemma3:4b' not found" in response

# Test: unexpected error handling
@patch("ollama.chat")
def test_chat_unexpected_error(mock_chat):
    mock_chat.side_effect = Exception("Internal server error")
    response = ChatHandler.chat("Hi!")
    assert "An unexpected error occurred with model 'gemma3:4b'" in response

# Test: Message model instantiation
def test_message_model_defaults():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.timestamp is None

def test_message_model_with_timestamp():
    now = datetime.now()
    msg = Message(role="system", content="Start", timestamp=now)
    assert msg.timestamp == now
