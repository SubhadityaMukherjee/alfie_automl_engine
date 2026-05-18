"""Tests for app.automlplus.website_accessibility.pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.automlplus.tools.text import ChunkResult
from app.automlplus.website_accessibility.pipeline import (
    resolve_coroutines,
    run_accessibility_pipeline,
    stream_accessibility_results,
)
from app.core.exceptions import AutoMLRuntimeError, AutoMLValidationError

# ---------------------------------------------------------------------------
# run_accessibility_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_content_returns_empty():
    result = await run_accessibility_pipeline(
        "", "test.html", MagicMock(), chunk_size=100
    )
    assert result == []


@pytest.mark.asyncio
async def test_whitespace_only_returns_empty():
    result = await run_accessibility_pipeline(
        "   \n\t  ", "test.html", MagicMock(), chunk_size=100
    )
    assert result == []


@pytest.mark.asyncio
async def test_invalid_chunk_size_raises():
    with pytest.raises(AutoMLValidationError, match="chunk_size must be > 0"):
        await run_accessibility_pipeline(
            "content", "test.html", MagicMock(), chunk_size=0
        )


@pytest.mark.asyncio
async def test_invalid_concurrency_raises():
    with pytest.raises(AutoMLValidationError, match="concurrency must be > 0"):
        await run_accessibility_pipeline(
            "content", "test.html", MagicMock(), chunk_size=10, concurrency=-1
        )


@pytest.mark.asyncio
@patch("app.automlplus.website_accessibility.pipeline.split_chunks")
async def test_split_failure_raises(mock_split):
    mock_split.side_effect = Exception("split failed")
    with pytest.raises(AutoMLRuntimeError, match="Failed to split content"):
        await run_accessibility_pipeline(
            "content", "test.html", MagicMock(), chunk_size=100
        )


@pytest.mark.asyncio
@patch("app.automlplus.website_accessibility.pipeline._process_single_chunk")
@patch(
    "app.automlplus.website_accessibility.pipeline.split_chunks",
    return_value=([], []),
)
async def test_no_chunks_returns_empty(mock_split, mock_process):
    result = await run_accessibility_pipeline(
        "content", "test.html", MagicMock(), chunk_size=100
    )
    assert result == []


@pytest.mark.asyncio
@patch("app.automlplus.website_accessibility.pipeline._process_single_chunk")
@patch(
    "app.automlplus.website_accessibility.pipeline.split_chunks",
    return_value=(["chunk1", "chunk2"], [(1, 10), (11, 20)]),
)
async def test_processes_chunks(mock_split, mock_process):
    chunk_results = [
        ChunkResult(
            chunk=0,
            start_line=1,
            end_line=10,
            score=85.0,
            image_feedback=[],
            llm_response="ok",
        ),
        ChunkResult(
            chunk=1,
            start_line=11,
            end_line=20,
            score=70.0,
            image_feedback=[],
            llm_response="ok2",
        ),
    ]
    mock_process.side_effect = chunk_results

    result = await run_accessibility_pipeline(
        "content", "test.html", MagicMock(), chunk_size=100
    )
    assert len(result) == 2
    assert result[0].score == 85.0
    assert result[1].score == 70.0


@pytest.mark.asyncio
@patch("app.automlplus.website_accessibility.pipeline._process_single_chunk")
@patch(
    "app.automlplus.website_accessibility.pipeline.split_chunks",
    return_value=(["chunk1"], [(1, 10)]),
)
async def test_process_failure_raises(mock_split, mock_process):
    mock_process.side_effect = Exception("processing failed")
    with pytest.raises(AutoMLRuntimeError, match="Failed to process chunks"):
        await run_accessibility_pipeline(
            "content", "test.html", MagicMock(), chunk_size=100
        )


# ---------------------------------------------------------------------------
# resolve_coroutines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_plain_value():
    assert await resolve_coroutines(42) == 42
    assert await resolve_coroutines("hello") == "hello"
    assert await resolve_coroutines(None) is None


@pytest.mark.asyncio
async def test_resolve_dict():
    result = await resolve_coroutines({"a": 1, "b": "two"})
    assert result == {"a": 1, "b": "two"}


@pytest.mark.asyncio
async def test_resolve_list():
    result = await resolve_coroutines([1, "two", 3.0])
    assert result == [1, "two", 3.0]


@pytest.mark.asyncio
async def test_resolve_object():
    class Foo:
        def __init__(self):
            self.x = 10
            self.y = "bar"

    result = await resolve_coroutines(Foo())
    assert result == {"x": 10, "y": "bar"}


@pytest.mark.asyncio
async def test_resolve_coroutine():
    async def coro():
        return "resolved"

    result = await resolve_coroutines(coro())
    assert result == "resolved"


@pytest.mark.asyncio
async def test_resolve_nested():
    async def inner():
        return "done"

    data = {"items": [1, inner()], "nested": {"val": inner()}}
    result = await resolve_coroutines(data)
    assert result == {"items": [1, "done"], "nested": {"val": "done"}}


# ---------------------------------------------------------------------------
# stream_accessibility_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_results_yields_json():
    results = [
        ChunkResult(
            chunk=0,
            start_line=1,
            end_line=10,
            score=80.0,
            image_feedback=[],
            llm_response="ok",
        )
    ]
    chunks = []
    async for data in stream_accessibility_results(results):
        chunks.append(data)

    assert len(chunks) == 1
    parsed = json.loads(chunks[0])
    assert len(parsed) == 1
    assert parsed[0]["score"] == 80.0


@pytest.mark.asyncio
async def test_stream_results_handles_coroutine_items():
    async def make_coro():
        return ChunkResult(
            chunk=0,
            start_line=1,
            end_line=5,
            score=50.0,
            image_feedback=[],
            llm_response="text",
        )

    chunks = []
    async for data in stream_accessibility_results([make_coro()]):
        chunks.append(data)

    parsed = json.loads(chunks[0])
    assert parsed[0]["score"] == 50.0


@pytest.mark.asyncio
async def test_stream_results_handles_coroutine_error():
    async def bad_coro():
        raise ValueError("bad")

    chunks = []
    async for data in stream_accessibility_results([bad_coro()]):
        chunks.append(data)

    parsed = json.loads(chunks[0])
    assert "error" in parsed[0]
