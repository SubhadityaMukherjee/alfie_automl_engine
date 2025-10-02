import pytest
from app.automlplus.imagetools import ImagePromptRunner
from unittest.mock import MagicMock, patch

@pytest.fixture
def jinja_template_example():
    return "this is {content}"

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
    messages = ImagePromptRunner.build_messages(jinja_environment= None, image_b64 = image_b64, prompt = prompt)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == prompt
    assert messages[0]["images"] == [image_b64]

def test_build_messages_with_jinja(jinja_template_example):
    prompt = "prompt"
    image_b64 = "image"
    messages = ImagePromptRunner.build_messages(jinja_environment= jinja_template_example, image_b64 = image_b64, prompt = prompt)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == prompt
    assert messages[0]["images"] == [image_b64]




# def test_build_messages_with_jinja():
#     mock_env = Mock()
#     # Mock render_template to return a fake system prompt
#     from your_module import render_template
#     render_template_backup = render_template
#     try:
#         def fake_render(env, template_name):
#             return "system prompt"
#         globals()['render_template'] = fake_render
#
#         image_b64 = "fake_base64"
#         prompt = "Describe this image"
#         messages = ImagePromptRunner.build_messages(mock_env, image_b64, prompt)
#         assert messages[0]["role"] == "system"
#         assert messages[0]["content"] == "system prompt"
#         assert messages[1]["role"] == "user"
#     finally:
#         globals()['render_template'] = render_template_backup
#
# def test_run_with_bytes():
#     fake_bytes = b"fake image bytes"
#     prompt = "Describe this image"
#
#     with patch("your_module.ImageConverter.bytes_to_base64", return_value="b64img") as mock_conv, \
#          patch("your_module.ChatHandler.chat_sync_messages", return_value="result") as mock_chat:
#
#         result = ImagePromptRunner.run(image_bytes=fake_bytes, prompt=prompt)
#
#         mock_conv.assert_called_once_with(fake_bytes)
#         mock_chat.assert_called_once()
#         assert result == "result"
#
#
# def test_run_stream():
#     fake_bytes = b"fake image bytes"
#     prompt = "Describe this image"
#
#     def fake_stream(messages, model):
#         yield "chunk1"
#         yield "chunk2"
#
#     with patch("your_module.ImageConverter.bytes_to_base64", return_value="b64img"), \
#          patch("your_module.ChatHandler.chat_stream_messages_sync", side_effect=fake_stream):
#
#         chunks = list(ImagePromptRunner.run_stream(image_bytes=fake_bytes, prompt=prompt))
#         assert chunks == ["chunk1", "chunk2"]
