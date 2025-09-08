import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, cast

from bs4 import BeautifulSoup

from app.core.chat_handler import ChatHandler
from app.core.utils import render_template
from app.website_accessibility.modules import AltTextChecker, split_chunks


@dataclass
class ChunkResult:
    chunk: int
    start_line: int
    end_line: int
    score: float | None
    image_feedback: List[Dict[str, Any]]
    llm_response: str | None
    error: str | None = None


def extract_text_from_html_bytes(content: bytes) -> str:
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
) -> ChunkResult:
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

            response_raw = await ChatHandler.chat(prompt, context="", stream=False)
            response_text = response_raw if isinstance(response_raw, str) else ""

            score_match = re.search(
                r"\bScore[:\s]*([0-9]+(?:\.[0-9]+)?)", response_text, re.IGNORECASE
            )
            score = float(score_match.group(1)) if score_match else None

            images = re.findall(r'<img[^>]+src="([^"]+)"[^>]*alt="([^"]+)"', chunk)
            image_feedback: List[Dict[str, Any]] = []
            for src, alt in images:
                try:
                    result = AltTextChecker.check(jinja_environment, src, alt)
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
) -> List[ChunkResult]:
    chunks, ranges = split_chunks(content, chunk_size)
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _process_single_chunk(
            i, chunk, start, end, len(chunks), filename, jinja_environment, sem
        )
        for i, (chunk, (start, end)) in enumerate(zip(chunks, ranges))
    ]
    results: List[ChunkResult] = await asyncio.gather(*tasks)
    return results


async def stream_accessibility_results(
    results: List[ChunkResult],
) -> AsyncGenerator[bytes, None]:
    scores = [r.score for r in results if r.score is not None]
    for r in results:
        yield (json.dumps(r.__dict__) + "\n").encode("utf-8")
    avg_score = round(sum(scores) / len(scores), 2) if scores else None
    yield (json.dumps({"average_score": avg_score}) + "\n").encode("utf-8")
