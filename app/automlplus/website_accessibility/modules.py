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
    """Check whether provided alt text matches an image using an LLM/VLM."""

    DEFAULT_MODEL = "qwen2.5vl"

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
        # Normalize common Azure GPT-4o-mini aliases (e.g., 'gpt40-mini', 'gpt4o-mini')
        lower = candidate.lower().replace(" ", "")
        if lower in {"gpt40-mini", "gpt4o-mini"}:
            return "gpt-4o-mini"
        return candidate

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
                images = message["images"]
                safe_images = []
                for img in images:
                    try:
                        length_hint = len(img) if isinstance(img, str) else None
                    except Exception:
                        length_hint = None
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
        model: str = os.getenv("ALT_TEXT_CHECKER_MODEL", "qwen2.5vl"),
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

            backend = os.getenv("MODEL_BACKEND", "ollama").lower()
            # Choose a sane default model per backend if caller/env leaves default
            if (not model or not model.strip()) and backend == "azure":
                model = "gpt-4o-mini"
            logger.info("Sending request to %s with model: %s", backend, model)
            # Redact base64 image data from logs to avoid printing large sensitive payloads
            logger.info(
                "Messages structure (redacted): %s",
                AltTextChecker._redact_messages_for_log(messages),
            )

            response_content = ChatHandler.chat_sync_messages(
                messages=messages,
                backend=backend,
                model=model,
            )
            return response_content
        except Exception as e:
            logger.exception("AltTextChecker failed with error: %s", str(e))
            logger.error("Model used: %s", model)
            # Log redacted messages on error as well
            try:
                logger.error(
                    "Messages sent (redacted): %s",
                    AltTextChecker._redact_messages_for_log(messages),
                )
            except Exception:
                logger.error("Messages sent (redaction_failed)")
            raise e


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


def split_chunks(content: str, chunk_size: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Split long content into chunks and track line ranges for each chunk."""
    lines = content.splitlines()
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    chunks, line_ranges, i = [], [], 0
    while i < len(content):
        end = i + chunk_size
        chunks.append(content[i:end])
        start_line = next(j for j, offset in enumerate(line_offsets) if offset > i) - 1
        end_line = (
            next(
                (j for j, offset in enumerate(line_offsets) if offset > end), len(lines)
            )
            - 1
        )
        line_ranges.append((start_line + 1, end_line + 1))
        i = end
    return chunks, line_ranges
