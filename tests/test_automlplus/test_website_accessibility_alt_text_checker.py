import logging
import os
from unittest.mock import patch

import pytest
from dotenv import find_dotenv, load_dotenv
from jinja2 import Environment, FileSystemLoader

from app.automlplus.tools.vlm import AltTextChecker

load_dotenv(find_dotenv())

jinja_path = os.getenv("JINJAPATH", "app/core/prompt_templates")


logger = logging.getLogger(__name__)


def test_alt_text_checker_default_model():
    assert AltTextChecker.DEFAULT_MODEL == "gpt-4o-mini"


def test_alt_text_checker_resolve_model_no_model(model=""):
    assert AltTextChecker._resolve_model(model=model) == AltTextChecker.DEFAULT_MODEL


def test_alt_text_checker_resolve_model_model_name(model="HelloIAm SmartModel1"):
    assert AltTextChecker._resolve_model(model=model) == "helloiamsmartmodel1"


def test_alt_text_checker_resolve_model_gpt_name_fixes():
    assert AltTextChecker._resolve_model(model="gpt40-mini") == "gpt-4o-mini"
    assert AltTextChecker._resolve_model(model="gpt4o-mini") == "gpt-4o-mini"
    assert AltTextChecker._resolve_model(model="gpt-4o-mini") == "gpt-4o-mini"


@pytest.fixture
def jinja_template_example():
    if jinja_path is not None:
        return Environment(loader=FileSystemLoader(jinja_path))


@pytest.fixture
def fake_image_path():
    return "ehe.png"


@pytest.fixture
def fake_image_bytes():
    return b"fake_image_bytes"


@pytest.fixture
def fake_image_b64():
    return "ZmFrZV9pbWFnZQ=="  # "fake_image" in base64


@pytest.fixture
def fake_alt_text():
    return "I am fake text"


@pytest.fixture
def fake_llm_text():
    return "The alt text is wonderfully generated"


@pytest.fixture
def fake_message():
    return [
        {
            "role": "system",
            "content": "You are a WCAG accessibility checker. Your job is to determine if the alt text meaningfully and accurately represents the image",
        },
        {"role": "user", "content": "Alt text: I am fake text"},
        {
            "role": "user",
            "content": "Does this alt text correctly describe the image? Respond with 'Yes' or 'No' and give a short justification.",
            "images": [
                "ZmFrZV9pbWFnZQ==dhfksdhgkllfslysfjhgkjsdkjfhsdlh08842kjhfgkjdshkgjsd"
            ],
        },
    ]


def test_alt_text_checker_build_messages(
    jinja_template_example, fake_image_b64, fake_alt_text
):
    res = AltTextChecker._build_messages(
        jinja_environment=jinja_template_example,
        image_b64=fake_image_b64,
        alt_text=fake_alt_text,
    )
    assert res == [
        {
            "role": "system",
            "content": "You are a WCAG accessibility checker. Your job is to determine if the alt text meaningfully and accurately represents the image",
        },
        {"role": "user", "content": "Alt text: I am fake text"},
        {
            "role": "user",
            "content": "Does this alt text correctly describe the image? Respond with 'Yes' or 'No' and give a short justification.",
            "images": ["ZmFrZV9pbWFnZQ=="],
        },
    ]


def test_alt_text_checker_redact_messages_for_log(fake_message):
    res = AltTextChecker._redact_messages_for_log(fake_message)
    assert res == [
        {
            "role": "system",
            "content": "You are a WCAG accessibility checker. Your job is to determine if the alt text meaningfully and accurately represents the image",
        },
        {"role": "user", "content": "Alt text: I am fake text"},
        {
            "role": "user",
            "content": "Does this alt text correctly describe the image? Respond with 'Yes' or 'No' and give a short justification.",
            "images": ["<redacted_base64 length=68>"],
        },
    ]


def test_alt_text_checker_checker_raises_exception(
    jinja_template_example, fake_image_path
):
    with patch("app.automlplus.utils.ImageConverter.to_base64") as mock_b64:
        mock_b64.side_effect = RuntimeError("Image Failed")

        with pytest.raises(RuntimeError):
            AltTextChecker.check(jinja_template_example, fake_image_path, "eh")


def test_alt_text_checker_checker_works(
    jinja_template_example,
    fake_image_b64,
    fake_image_path,
    fake_alt_text,
    fake_llm_text,
):
    with (
        patch("app.automlplus.utils.ImageConverter.to_base64") as mock_b64,
        patch("app.core.chat_handler.ChatHandler.chat_sync_messages") as mock_chat,
    ):
        mock_b64.return_value = fake_image_b64
        mock_chat.return_value = fake_llm_text

        result = AltTextChecker.check(
            jinja_template_example, fake_image_path, fake_alt_text, model="gpt-4o-mini"
        )

        assert result == fake_llm_text

        mock_b64.assert_called_once_with(fake_image_path)
        mock_chat.assert_called_once()
