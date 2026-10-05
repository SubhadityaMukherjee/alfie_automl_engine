"""Tests for app.automlplus.tools.text (_process_single_chunk, _process_chunk_batch, ChunkResult, summarize)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.automlplus.tools.text import (
    ChunkResult,
    _parse_batch_sections,
    _process_chunk_batch,
    _process_single_chunk,
    summarize_accessibility_results,
)


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
    assert r.page is None
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
    assert "  " not in (result.llm_response or "")


# ---------------------------------------------------------------------------
# _process_chunk_batch
# ---------------------------------------------------------------------------


def _batch_item(i: int, chunk: str, start: int = 1, end: int = 10):
    return (i, chunk, start, end)


def _batch_response(*reviews: str) -> str:
    return " ".join(
        f"=== CHUNK {n + 1} REVIEW START === {review} === CHUNK {n + 1} REVIEW END ==="
        for n, review in enumerate(reviews)
    )


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_batch_singleton_uses_single_prompt(
    mock_render, mock_chat, mock_alt, mock_jinja, sem
):
    mock_chat.chat = AsyncMock(return_value="Score: 85.")

    results = await _process_chunk_batch(
        [_batch_item(0, "<p>Hello</p>")], 1, "test.html", mock_jinja, sem, ""
    )

    assert len(results) == 1
    assert results[0].score == 85.0
    assert mock_render.call_args.kwargs["template_name"] == "build_chunk_prompt.txt"
    mock_chat.chat.assert_awaited_once()


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_batch_sends_one_request_for_multiple_chunks(
    mock_render, mock_chat, mock_alt, mock_jinja, sem
):
    mock_chat.chat = AsyncMock(
        return_value=_batch_response("Score: 8. Good.", "Score: 5. Missing alt texts.")
    )

    batch = [_batch_item(0, "<p>One</p>"), _batch_item(1, "<p>Two</p>")]
    results = await _process_chunk_batch(batch, 2, "test.html", mock_jinja, sem, "")

    assert mock_render.call_args.kwargs["template_name"] == (
        "build_batched_chunk_prompt.txt"
    )
    assert mock_render.call_args.kwargs["num_chunks"] == 2
    mock_chat.chat.assert_awaited_once()
    assert len(results) == 2
    assert [r.chunk for r in results] == [0, 1]
    assert results[0].score == 8.0
    assert results[1].score == 5.0
    assert "Good." in (results[0].llm_response or "")
    assert "Missing alt texts." in (results[1].llm_response or "")


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_batch_missing_section_marks_chunk(
    mock_render, mock_chat, mock_alt, mock_jinja, sem
):
    mock_chat.chat = AsyncMock(return_value=_batch_response("Score: 8. Good."))

    batch = [_batch_item(0, "<p>One</p>"), _batch_item(1, "<p>Two</p>")]
    results = await _process_chunk_batch(batch, 2, "test.html", mock_jinja, sem, "")

    assert results[0].score == 8.0
    assert results[0].error is None
    assert results[1].score is None
    assert results[1].error == "No review section for chunk in batched response"


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_batch_error_marks_all_chunks(
    mock_render, mock_chat, mock_alt, mock_jinja, sem
):
    mock_chat.chat = AsyncMock(side_effect=Exception("LLM down"))

    batch = [_batch_item(0, "<p>One</p>"), _batch_item(1, "<p>Two</p>")]
    results = await _process_chunk_batch(batch, 2, "test.html", mock_jinja, sem, "")

    assert len(results) == 2
    assert all(r.error == "LLM down" for r in results)
    assert all(r.score is None for r in results)


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.AltTextChecker")
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="rendered prompt")
async def test_batch_skips_empty_chunks(
    mock_render, mock_chat, mock_alt, mock_jinja, sem
):
    # Only chunk index 1 is sent; the prompt labels it "chunk 2", so the
    # review section comes back numbered 2.
    mock_chat.chat = AsyncMock(
        return_value=(
            "=== CHUNK 2 REVIEW START === Score: 7. === CHUNK 2 REVIEW END ==="
        )
    )

    batch = [_batch_item(0, "   "), _batch_item(1, "<p>Two</p>")]
    results = await _process_chunk_batch(batch, 2, "test.html", mock_jinja, sem, "")

    assert results[0].error == "Empty chunk provided"
    assert mock_render.call_args.kwargs["num_chunks"] == 1
    assert results[1].score == 7.0


def test_parse_batch_sections_numbers_and_normalization():
    response = (
        "=== CHUNK 2 REVIEW START ===  Score:   9.  Great   === CHUNK 2 REVIEW END === "
        "=== CHUNK 1 REVIEW START === Score: 4 === CHUNK 1 REVIEW END ==="
    )
    sections = _parse_batch_sections(response)
    assert sections[2] == "Score: 9. Great"
    assert sections[1] == "Score: 4"


def test_parse_batch_sections_no_matches():
    assert _parse_batch_sections("no sections here") == {}


# ---------------------------------------------------------------------------
# summarize_accessibility_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="summary prompt")
async def test_summarize_returns_llm_text(mock_render, mock_chat, mock_jinja):
    mock_chat.chat = AsyncMock(return_value="## Overall\nGood site")

    summary = await summarize_accessibility_results(
        jinja_environment=mock_jinja,
        source="https://example.com",
        pages=["https://example.com"],
        average_score=6.5,
        chunk_scores=[{"page": "https://example.com", "chunk": 0, "score": 6.5}],
        readability={"Flesch Reading Ease": 25.7},
        llm_responses=["Score: 6.5 missing alt texts"],
    )
    assert summary == "## Overall\nGood site"
    mock_chat.chat.assert_awaited_once()


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template")
async def test_summarize_truncates_long_findings(mock_render, mock_chat, mock_jinja):
    mock_chat.chat = AsyncMock(return_value="summary")
    mock_render.return_value = "prompt"

    await summarize_accessibility_results(
        jinja_environment=mock_jinja,
        source="s",
        pages=["s"],
        average_score=None,
        chunk_scores=[],
        readability=None,
        llm_responses=["x" * 5000],
    )

    rendered = mock_render.call_args.kwargs
    assert len(rendered["findings"]) < 1000


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="prompt")
async def test_summarize_no_findings_uses_placeholder(
    mock_render, mock_chat, mock_jinja
):
    mock_chat.chat = AsyncMock(return_value="summary")

    await summarize_accessibility_results(
        jinja_environment=mock_jinja,
        source="s",
        pages=["s"],
        average_score=None,
        chunk_scores=[],
        readability=None,
        llm_responses=[""],
    )

    rendered = mock_render.call_args.kwargs
    assert "No chunk findings available." in rendered["findings"]


@pytest.mark.asyncio
@patch("app.automlplus.tools.text.ChatHandler")
@patch("app.automlplus.tools.text.render_template", return_value="prompt")
async def test_summarize_non_string_response_returns_empty(
    mock_render, mock_chat, mock_jinja
):
    mock_chat.chat = AsyncMock(return_value=42)

    summary = await summarize_accessibility_results(
        jinja_environment=mock_jinja,
        source="s",
        pages=["s"],
        average_score=None,
        chunk_scores=[],
        readability=None,
        llm_responses=[],
    )
    assert summary == ""
