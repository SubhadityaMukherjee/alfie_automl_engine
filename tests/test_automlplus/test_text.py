"""Tests for app.automlplus.tools.text (_process_single_chunk, ChunkResult)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.automlplus.tools.text import ChunkResult, _process_single_chunk


@pytest.fixture
def mock_jinja():
    return MagicMock()


@pytest.fixture
def sem():
    return asyncio.Semaphore(4)


# ---------------------------------------------------------------------------
# ChunkResult defaults
# ---------------------------------------------------------------------------


def test_chunk_result_defaults():
    r = ChunkResult(
        chunk=0,
        start_line=1,
        end_line=10,
        score=85.0,
        image_feedback=[],
        llm_response="ok",
    )
    assert r.error is None
    assert r.chunk == 0
    assert r.score == 85.0


# ---------------------------------------------------------------------------
# _process_single_chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_chunk(mock_jinja, sem):
    result = await _process_single_chunk(
        0, "", 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.error == "Empty chunk provided"
    assert result.score is None


@pytest.mark.asyncio
async def test_whitespace_chunk(mock_jinja, sem):
    result = await _process_single_chunk(
        0, "   \n\t  ", 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.error == "Empty chunk provided"


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_success_with_score(mock_render, mock_chat, mock_alt, mock_jinja, sem):
    mock_chat.chat = AsyncMock(return_value="Score: 85. Some feedback text.")
    mock_alt.check.return_value = "Alt text is good"

    result = await _process_single_chunk(
        0, "<p>Hello</p>", 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.score == 85.0
    assert result.error is None
    assert result.llm_response == "Score: 85. Some feedback text."


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_score_out_of_range(mock_render, mock_chat, mock_alt, mock_jinja, sem):
    mock_chat.chat = AsyncMock(return_value="Score: 150.")

    result = await _process_single_chunk(
        0, "<p>Hello</p>", 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.score is None
    assert result.error is None


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_no_score_in_response(mock_render, mock_chat, mock_alt, mock_jinja, sem):
    mock_chat.chat = AsyncMock(return_value="No score here, just text.")

    result = await _process_single_chunk(
        0, "<p>Hello</p>", 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.score is None
    assert result.llm_response == "No score here, just text."


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_with_images(mock_render, mock_chat, mock_alt, mock_jinja, sem):
    mock_chat.chat = AsyncMock(return_value="Score: 70.")
    mock_alt.check.return_value = "Good alt text"

    html = '<img src="http://example.com/img.png" alt="A picture"><p>Text</p>'
    result = await _process_single_chunk(
        0, html, 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.score == 70.0
    assert len(result.image_feedback) == 1
    assert result.image_feedback[0]["src"] == "http://example.com/img.png"
    assert result.image_feedback[0]["alt_text"] == "A picture"


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_image_check_error(mock_render, mock_chat, mock_alt, mock_jinja, sem):
    mock_chat.chat = AsyncMock(return_value="Score: 60.")
    mock_alt.check.side_effect = RuntimeError("check failed")

    html = '<img src="http://example.com/img.png" alt="desc">'
    result = await _process_single_chunk(
        0, html, 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert len(result.image_feedback) == 1
    assert "error" in result.image_feedback[0]
    assert "check failed" in result.image_feedback[0]["error"]


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch(
    "app.automlplus.tools.text.render_template",
    side_effect=RuntimeError("template fail"),
)
async def test_template_render_error(mock_render, mock_chat, mock_alt, mock_jinja, sem):
    result = await _process_single_chunk(
        0, "<p>Hello</p>", 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.error is not None
    assert "template fail" in result.error


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="prompt")
async def test_chat_error(mock_render, mock_chat, mock_alt, mock_jinja, sem):
    mock_chat.chat = AsyncMock(side_effect=Exception("LLM down"))

    result = await _process_single_chunk(
        0, "<p>Hello</p>", 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.error is not None
    assert "LLM down" in result.error


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="prompt")
async def test_custom_model_env(mock_render, mock_chat, mock_alt, mock_jinja, sem):
    mock_chat.chat = AsyncMock(return_value="Score: 90.")

    with patch.dict("os.environ", {"WEB_ACCESSIBILITY_CHAT_MODEL": "custom-model "}):
        await _process_single_chunk(
            0, "<p>Hello</p>", 1, 10, 1, "test.html", mock_jinja, sem, ""
        )
        call_kwargs = mock_chat.chat.call_args
        assert call_kwargs[1]["model"] == "custom-model"


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="prompt")
async def test_whitespace_normalization(
    mock_render, mock_chat, mock_alt, mock_jinja, sem
):
    mock_chat.chat = AsyncMock(return_value="  Score:   75.  Lots   of   spaces  ")

    result = await _process_single_chunk(
        0, "<p>Hello</p>", 1, 10, 1, "test.html", mock_jinja, sem, ""
    )
    assert result.score == 75.0
    assert "  " not in result.llm_response
