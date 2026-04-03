"""Orchestration pipeline for web accessibility analysis.

This module coordinates the full accessibility analysis workflow: it splits an
HTML document into chunks, fans out concurrent LLM-over-text analysis via
``_process_single_chunk``, and aggregates results. It is intentionally thin —
all tool logic lives in ``app.automlplus.tools``.

- ``run_accessibility_pipeline`` — main entry point; returns a list of
  ``ChunkResult`` objects, one per chunk.
- ``resolve_coroutines`` — utility to recursively await coroutine-valued
  attributes when serialising results.
- ``stream_accessibility_results`` — streams the resolved results as a single
  JSON array (used for streaming response endpoints).
"""

import asyncio
import json
import logging
from typing import Any, List

from app.automlplus.tools.static import split_chunks
from app.automlplus.tools.text import ChunkResult, _process_single_chunk

logger = logging.getLogger(__name__)


async def run_accessibility_pipeline(
    content: str,
    filename: str,
    jinja_environment,
    chunk_size: int,
    concurrency: int = 4,
    context: str = "",
) -> List[ChunkResult]:
    """Split HTML into chunks and process them concurrently with a semaphore."""
    if not content or not content.strip():
        logger.warning("Empty content provided to run_accessibility_pipeline")
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    if concurrency <= 0:
        raise ValueError(f"concurrency must be > 0, got {concurrency}")

    try:
        chunks, ranges = split_chunks(content, chunk_size)
    except Exception as e:
        logger.exception("Failed to split content into chunks")
        raise RuntimeError(f"Failed to split content into chunks: {e}") from e

    logger.info("Processing the website in %d chunks", len(chunks))

    if not chunks:
        logger.warning("No chunks generated from content")
        return []

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _process_single_chunk(
            i, chunk, start, end, len(chunks), filename, jinja_environment, sem, context
        )
        for i, (chunk, (start, end)) in enumerate(zip(chunks, ranges))
    ]

    try:
        results: List[ChunkResult] = await asyncio.gather(
            *tasks, return_exceptions=False
        )
    except Exception as e:
        logger.exception("Failed to process chunks")
        raise RuntimeError(f"Failed to process chunks: {e}") from e

    return results


async def resolve_coroutines(obj: Any) -> Any:
    """Recursively await any coroutine attributes in an object."""
    if asyncio.iscoroutine(obj):
        return await obj
    elif isinstance(obj, dict):
        return {k: await resolve_coroutines(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [await resolve_coroutines(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        new_obj = {}
        for k, v in vars(obj).items():
            new_obj[k] = await resolve_coroutines(v)
        return new_obj
    else:
        return obj


async def stream_accessibility_results(results):
    """Stream results as a single JSON array instead of JSONL."""
    resolved = []
    for item in results:
        if asyncio.iscoroutine(item):
            try:
                item = await item
            except Exception as e:
                resolved.append({"error": str(e)})
                continue

        try:
            data = await resolve_coroutines(item)
        except Exception as e:
            data = {"error": f"Failed to resolve item: {e}"}

        resolved.append(data)

    yield json.dumps(resolved, indent=2).encode("utf-8")
