"""LLM-over-text tools for AutoML+.

Text tools send plain-text content (HTML chunks, documents, etc.) to a language
model and parse the structured response. Unlike VLM tools, no image input is
required; unlike static tools, they rely on an external LLM API.

Current tools:

- ``ChunkResult`` — dataclass holding the outcome (score, image feedback, LLM
  response, or error) for a single processed text chunk.
- ``_process_single_chunk`` — sends one HTML chunk to the LLM for WCAG analysis,
  extracts a numeric score from the response, and runs ``AltTextChecker`` on any
  ``<img>`` tags found in the chunk.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from app.automlplus.tools.vlm import AltTextChecker
from app.core.chat_handler import ChatHandler
from app.core.config import get_settings
from app.core.utils import render_template

logger = logging.getLogger(__name__)


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
            if not chunk or not chunk.strip():
                logger.warning("Empty chunk provided for processing at index %d", i)
                return ChunkResult(
                    chunk=i,
                    start_line=start,
                    end_line=end,
                    score=None,
                    image_feedback=[],
                    llm_response=None,
                    error="Empty chunk provided",
                )

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

            settings = get_settings()
            backend = settings.model_backend.lower()
            model = (
                settings.web_accessibility_chat_model or ""
            ).strip() or "gpt-4o-mini"
            response_raw = await ChatHandler.chat(
                prompt,
                context=context,
                backend=backend,
                model=model,
                stream=False,
            )
            response_text = response_raw if isinstance(response_raw, str) else ""

            def _normalize_text(text: str) -> str:
                return re.sub(r"\s+", " ", text).strip()

            response_text = _normalize_text(response_text)

            score_match = re.search(
                r"\bScore[:\s]*([0-9]+(?:\.[0-9]+)?)", response_text, re.IGNORECASE
            )
            score = None
            if score_match:
                try:
                    score_val = float(score_match.group(1))
                    if 0 <= score_val <= 100:
                        score = score_val
                    else:
                        logger.warning(
                            "Score %f out of valid range [0, 100] for chunk %d",
                            score_val,
                            i,
                        )
                except ValueError as e:
                    logger.warning(
                        "Failed to parse score '%s' for chunk %d: %s",
                        score_match.group(1),
                        i,
                        e,
                    )

            images = re.findall(r'<img[^>]+src="([^"]+)"[^>]*alt="([^"]+)"', chunk)
            image_feedback: List[Dict[str, Any]] = []
            for src, alt in images:
                try:
                    result = AltTextChecker.check(jinja_environment, src, alt)
                    if isinstance(result, str):
                        result = _normalize_text(result)
                    image_feedback.append(
                        {"src": src, "alt_text": alt, "result": result}
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to check alt text for image '%s' in chunk %d: %s",
                        src,
                        i,
                        e,
                    )
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
        except Exception as e:
            logger.exception("Failed to process chunk %d", i)
            return ChunkResult(
                chunk=i,
                start_line=start,
                end_line=end,
                score=None,
                image_feedback=[],
                llm_response=None,
                error=str(e),
            )
