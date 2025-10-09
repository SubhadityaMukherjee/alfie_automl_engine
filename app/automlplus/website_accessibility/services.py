import asyncio
import json
import re
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from bs4 import BeautifulSoup

from app.automlplus.website_accessibility.modules import AltTextChecker, split_chunks
from app.core.chat_handler import ChatHandler
from app.core.utils import render_template


@dataclass
class ChunkResult:
    """Result for processing a single chunk of an HTML file."""

    chunk: int
    start_line: int
    end_line: int
    score: float | None
    image_feedback: List[Dict[str, Any]]
    llm_response: str | None
    error: str | None = None


def extract_text_from_html_bytes(content: bytes) -> str:
    """Extract readable text from raw HTML bytes."""
    soup = BeautifulSoup(content, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    lines = (line.strip() for line in soup.get_text().splitlines())
    phrases = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in phrases if chunk)
    return text


async def _process_single_chunk(
    i: int,
    chunk: str,
    start: int,
    end: int,
    total: int,
    filename: str,
    jinja_environment,
    sem: asyncio.Semaphore,
    context: str,
) -> ChunkResult:
    """Process a single chunk: prompt LLM and validate image alt texts."""
    async with sem:
        try:
            prompt = render_template(
                jinja_environment=jinja_environment,
                template_name="build_chunk_prompt.txt",
                filename=filename,
                chunk=chunk,
                idx=i,
                total=total,
                start_line=start,
                end_line=end,
            )

            backend = os.getenv("MODEL_BACKEND", "ollama").lower()
            model_env = os.getenv("WEB_ACCESSIBILITY_CHAT_MODEL")
            # Default model depends on backend
            if model_env and model_env.strip():
                model = model_env
            else:
                model = "gpt-4o-mini" if backend == "azure" else "gemma3:4b"
            response_raw = await ChatHandler.chat(
                prompt,
                context=context,
                backend=backend,
                model=model,
                stream=False,
            )
            response_text = response_raw if isinstance(response_raw, str) else ""
            # Normalize LLM text to avoid embedded newlines and odd whitespace sequences in JSON
            def _normalize_text(text: str) -> str:
                # Collapse all whitespace (including newlines, tabs) to single spaces
                return re.sub(r"\s+", " ", text).strip()

            response_text = _normalize_text(response_text)

            score_match = re.search(
                r"\bScore[:\s]*([0-9]+(?:\.[0-9]+)?)", response_text, re.IGNORECASE
            )
            score = float(score_match.group(1)) if score_match else None

            images = re.findall(r'<img[^>]+src="([^"]+)"[^>]*alt="([^"]+)"', chunk)
            image_feedback: List[Dict[str, Any]] = []
            for src, alt in images:
                try:
                    result = AltTextChecker.check(jinja_environment, src, alt)
                    # Ensure alt-text evaluation is a compact, JSON-friendly string
                    if isinstance(result, str):
                        result = _normalize_text(result)
                    image_feedback.append(
                        {"src": src, "alt_text": alt, "result": result}
                    )
                except Exception as e:  # noqa: BLE001 - surface image-level errors
                    image_feedback.append(
                        {"src": src, "alt_text": alt, "error": str(e)}
                    )

            return ChunkResult(
                chunk=i,
                start_line=start,
                end_line=end,
                score=score,
                image_feedback=image_feedback,
                llm_response=response_text,
                error=None,
            )
        except Exception as e:  # noqa: BLE001 - capture per-chunk failures
            return ChunkResult(
                chunk=i,
                start_line=start,
                end_line=end,
                score=None,
                image_feedback=[],
                llm_response=None,
                error=str(e),
            )


async def run_accessibility_pipeline(
    content: str,
    filename: str,
    jinja_environment,
    chunk_size: int,
    concurrency: int = 4,
    context: str = "",
) -> List[ChunkResult]:
    """Split HTML into chunks and process them concurrently with a semaphore."""
    chunks, ranges = split_chunks(content, chunk_size)
    import logging
    logging.getLogger(__name__).info("Processing the website in %d chunks", len(chunks))
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _process_single_chunk(
            i, chunk, start, end, len(chunks), filename, jinja_environment, sem, context
        )
        for i, (chunk, (start, end)) in enumerate(zip(chunks, ranges))
    ]
    results: List[ChunkResult] = await asyncio.gather(*tasks)
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
        # Make a copy of the object's dict with coroutines resolved
        new_obj = {}
        for k, v in vars(obj).items():
            new_obj[k] = await resolve_coroutines(v)
        return new_obj
    else:
        return obj

async def stream_accessibility_results(results):
    """
    Stream results as a single JSON array instead of JSONL.
    """
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
