import bisect
import logging
import os
from math import isfinite
from typing import Any, Dict, List, Tuple

import textstat  # type: ignore
from jinja2 import Environment  # type: ignore

from app.automlplus.utils import ImageConverter
from app.core.chat_handler import ChatHandler
from app.core.utils import render_template

logger = logging.getLogger(__name__)


class AltTextChecker:
    """Check whether provided alt text matches an image using a VLM."""

    DEFAULT_MODEL = "gpt-4o-mini"

    @staticmethod
    def _resolve_model(model: str) -> str:
        """Return a valid model string, normalizing common aliases and falling back to default."""
        if not model or model.strip() == "":
            logger.error(
                "Model parameter is empty or None, using default '%s'",
                AltTextChecker.DEFAULT_MODEL,
            )
            return AltTextChecker.DEFAULT_MODEL

        candidate = model.strip()
        lower = candidate.lower().replace(" ", "")

        # Normalize common GPT-4o mini aliases
        if lower in {"gpt40-mini", "gpt4o-mini"}:
            return "gpt-4o-mini"

        return lower

    @staticmethod
    def _build_messages(
        jinja_environment: Environment, image_b64: str, alt_text: str
    ) -> List[dict]:
        """Construct the message payload for the VLM call."""
        return [
            {
                "role": "system",
                "content": render_template(
                    jinja_environment, "wcag_checker_default_prompt.txt"
                ),
            },
            {"role": "user", "content": f"Alt text: {alt_text}"},
            {
                "role": "user",
                "content": render_template(
                    jinja_environment, "image_alt_checker_prompt.txt"
                ),
                "images": [image_b64],
            },
        ]

    @staticmethod
    def _redact_messages_for_log(messages: List[dict]) -> List[dict]:
        """Return a copy of messages with any base64 image payloads redacted for logging."""
        redacted: List[dict] = []
        for message in messages:
            msg_copy = {k: v for k, v in message.items() if k != "images"}
            if "images" in message:
                safe_images = []
                for img in message["images"]:
                    length_hint = len(img) if isinstance(img, str) else None
                    safe_images.append(
                        f"<redacted_base64 length={length_hint}>"
                        if length_hint is not None
                        else "<redacted_base64>"
                    )
                msg_copy["images"] = safe_images
            redacted.append(msg_copy)
        return redacted

    @staticmethod
    def check(
        jinja_environment: Environment,
        image_url_or_path: str,
        alt_text: str,
        model: str = os.getenv("ALT_TEXT_CHECKER_MODEL", DEFAULT_MODEL),
    ) -> str:
        logger.info("Checking alt-text using model %s", model)
        model = AltTextChecker._resolve_model(model)

        try:
            image_b64 = ImageConverter.to_base64(image_url_or_path)

            messages = AltTextChecker._build_messages(
                jinja_environment=jinja_environment,
                image_b64=image_b64,
                alt_text=alt_text,
            )

            logger.info("Sending request with model: %s", model)
            logger.info(
                "Messages structure (redacted): %s",
                AltTextChecker._redact_messages_for_log(messages),
            )

            response_content = ChatHandler.chat_sync_messages(
                messages=messages,
                model=model,
            )

            return response_content

        except Exception as e:
            logger.exception("AltTextChecker failed with error: %s", str(e))
            logger.error("Model used: %s", model)
            try:
                logger.error(
                    "Messages sent (redacted): %s",
                    AltTextChecker._redact_messages_for_log(messages),
                )
            except Exception:
                logger.error("Messages sent (redaction_failed)")
            raise


class ReadabilityAnalyzer:
    """Compute readability metrics for a piece of text."""

    METRICS = {
        "Flesch Reading Ease": textstat.flesch_reading_ease,
        "Difficult Words": textstat.difficult_words,
        "Lexicon Count": textstat.lexicon_count,
        "Avg Sentence Length": textstat.words_per_sentence,
    }

    @staticmethod
    def apply_metric(metric, text: str) -> Any:
        try:
            value = metric(text)
            # Normalize to JSON-serializable primitives and avoid NaN/Infinity
            if isinstance(value, float) and not isfinite(value):
                return None
            if isinstance(value, (int, float, str)):
                return value
            return str(value)
        except Exception:
            logger.warning("Metric failed: %s", metric.__name__)
            return "N/A"

    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        logger.info("Running readability metrics")
        return {
            name: cls.apply_metric(metric, text) for name, metric in cls.METRICS.items()
        }


def split_chunks(
    content: str, chunk_size: int
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Split content into fixed-size character chunks and return
    1-based (start_line, end_line) ranges for each chunk.

    Line ranges are accurate even when chunks start/end mid-line.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Keep newlines so offsets are exact
    lines = content.splitlines(keepends=True)

    # Build cumulative character offsets
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    chunks: List[str] = []
    line_ranges: List[Tuple[int, int]] = []

    content_len = len(content)
    i = 0

    while i < content_len:
        end = min(i + chunk_size, content_len)
        chunks.append(content[i:end])

        # Find line indices via binary search
        start_line = bisect.bisect_right(line_offsets, i) - 1
        end_line = bisect.bisect_left(line_offsets, end) - 1

        # Convert to 1-based inclusive line numbers
        line_ranges.append((start_line + 1, max(start_line + 1, end_line + 1)))

        i = end

    return chunks, line_ranges
