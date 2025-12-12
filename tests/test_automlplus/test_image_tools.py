import os
from unittest.mock import patch

import pytest
from dotenv import find_dotenv, load_dotenv
from jinja2 import Environment, FileSystemLoader

from app.automlplus.imagetools import ImagePromptRunner
from app.automlplus.utils import ImageConverter
from app.core.chat_handler import ChatHandler

load_dotenv(find_dotenv())

jinja_path = os.getenv("JINJAPATH")
if not jinja_path:
    raise RuntimeError("JINJAPATH environment variable is not set")


@pytest.fixture
def fake_image_bytes():
    return b"fake_image_bytes"


@pytest.fixture
def fake_image_b64():
    return "ZmFrZV9pbWFnZQ=="  # "fake_image" in base64


@pytest.fixture
def jinja_template_example():
    return Environment(loader=FileSystemLoader(jinja_path))


def test_default_model_name():
    assert isinstance(ImagePromptRunner.DEFAULT_MODEL, str)


def test_resolve_model():
    default = ImagePromptRunner.DEFAULT_MODEL

    assert ImagePromptRunner._resolve_model(None) == default
    assert ImagePromptRunner._resolve_model("") == default
    assert ImagePromptRunner._resolve_model("  ") == default
    assert ImagePromptRunner._resolve_model("custom_model") == "custom_model"


def test_build_messages_with_no_template():
    prompt = "prompt"
    image_b64 = "image"
    messages = ImagePromptRunner.build_messages(
        jinja_environment=None, image_b64=image_b64, prompt=prompt
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == prompt
    assert messages[0]["images"] == [image_b64]


def test_build_messages_with_jinja(jinja_template_example):
    prompt = "prompt"
    image_b64 = "image"
    messages = ImagePromptRunner.build_messages(
        jinja_environment=jinja_template_example, image_b64=image_b64, prompt=prompt
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == prompt
    assert messages[1]["images"] == [image_b64]


@patch.object(ChatHandler, "chat_sync_messages", return_value="ok_response")
@patch.object(ImageConverter, "bytes_to_base64", return_value="fake_b64")
def test_run_with_image_bytes(mock_to_b64, mock_chat):
    result = ImagePromptRunner.run(image_bytes=b"abc", prompt="describe this")
    assert result == "ok_response"
    mock_to_b64.assert_called_once_with(b"abc")
    mock_chat.assert_called_once()


@patch.object(ChatHandler, "chat_sync_messages", return_value="ok_response")
@patch.object(ImageConverter, "to_base64", return_value="fake_b64")
def test_run_with_image_path(mock_to_b64, mock_chat):
    result = ImagePromptRunner.run(
        image_path_or_url="path/to/image.png", prompt="describe this"
    )
    assert result == "ok_response"
    mock_to_b64.assert_called_once_with("path/to/image.png")
    mock_chat.assert_called_once()


def test_run_raises_value_error_when_no_input():
    with pytest.raises(
        ValueError, match="Provide either image_bytes or image_path_or_url"
    ):
        ImagePromptRunner.run(prompt="test")


@patch.object(ImageConverter, "bytes_to_base64", side_effect=RuntimeError("boom"))
def test_run_handles_conversion_error(mock_to_b64):
    with pytest.raises(RuntimeError, match="boom"):
        ImagePromptRunner.run(image_bytes=b"bad", prompt="oops")


@patch("app.automlplus.imagetools.logger")
@patch.object(ImageConverter, "to_base64", side_effect=ValueError("bad"))
def test_run_logs_exception(mock_to_b64, mock_logger):
    with pytest.raises(ValueError):
        ImagePromptRunner.run(image_path_or_url="x", prompt="desc")
    mock_logger.exception.assert_called()


@patch.object(ImagePromptRunner, "_resolve_model", return_value="mock_model")
@patch.object(ChatHandler, "chat_sync_messages", return_value="response")
@patch.object(ImageConverter, "bytes_to_base64", return_value="b64")
def test_run_calls_resolve_model(mock_b64, mock_chat, mock_resolve):
    ImagePromptRunner.run(image_bytes=b"x", prompt="desc")
    mock_resolve.assert_called_once()
    mock_chat.assert_called_once()


@patch.object(
    ChatHandler, "chat_stream_messages_sync", return_value=["chunk1", "chunk2"]
)
@patch.object(ImageConverter, "to_base64", return_value="fake_b64")
def test_run_stream_with_path(mock_to_b64, mock_stream):
    result = ImagePromptRunner.run_stream(
        image_path_or_url="url.png", prompt="caption this"
    )
    assert result == ["chunk1", "chunk2"]
    mock_to_b64.assert_called_once()
    mock_stream.assert_called_once()


@patch.object(ChatHandler, "chat_stream_messages_sync", return_value=["stream"])
@patch.object(ImageConverter, "bytes_to_base64", return_value="b64")
def test_run_stream_with_bytes(mock_b64, mock_stream):
    result = ImagePromptRunner.run_stream(image_bytes=b"x", prompt="hi")
    assert result == ["stream"]
    mock_b64.assert_called_once_with(b"x")
    mock_stream.assert_called_once()


def test_run_stream_raises_value_error_when_no_input():
    with pytest.raises(
        ValueError, match="Provide either image_bytes or image_path_or_url"
    ):
        ImagePromptRunner.run_stream(prompt="hi")
