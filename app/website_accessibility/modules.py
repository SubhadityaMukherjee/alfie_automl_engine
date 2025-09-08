import base64
import logging
import os
from io import BytesIO
from typing import Dict, List, Tuple

import requests
import textstat
from jinja2 import Environment
from ollama import Client
from PIL import Image
from urllib3.response import HTTPResponse

from app.core.utils import render_template

logger = logging.getLogger(__name__)

client = Client()
# Remove duplicate jinja environment creation - it will be passed from main.py
# jinja_path = os.getenv("JINJAPATH") or ""
# jinja_environment = Environment(loader=FileSystemLoader(Path(jinja_path)))


class ImageConverter:
    """Convert images to base64 from local paths or URLs."""

    @staticmethod
    def to_base64(image_path_or_url: str) -> str:
        logger.info("Converting image to base64: %s", image_path_or_url)
        try:

            if image_path_or_url.startswith("http"):
                raw_image: HTTPResponse = requests.get(
                    image_path_or_url, stream=True
                ).raw
                if raw_image is not None:
                    image = Image.open(raw_image)
            else:
                image = Image.open(image_path_or_url)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.exception("Image conversion failed")
            raise e


class AltTextChecker:
    """Check whether provided alt text matches an image using an LLM/VLM."""

    @staticmethod
    def check(
        jinja_environment: Environment,
        image_url_or_path: str,
        alt_text: str,
        model: str = os.getenv("ALT_TEXT_CHECKER_MODEL", "qwen2.5vl"),
    ) -> str:
        logger.info("Checking alt-text using model %s", model)

        # Validate model parameter
        if not model or model.strip() == "":
            logger.error("Model parameter is empty or None, using default 'qwen2.5vl'")
            model = "qwen2.5vl"

        try:
            image_b64 = ImageConverter.to_base64(image_url_or_path)
            messages = [
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

            logger.info("Sending request to ollama with model: %s", model)
            logger.info("Messages structure: %s", messages)

            response = client.chat(model=model, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            logger.exception("AltTextChecker failed with error: %s", str(e))
            logger.error("Model used: %s", model)
            logger.error("Messages sent: %s", messages)
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
    def apply_metric(metric, text: str) -> str:
        try:
            return metric(text)
        except Exception:
            logger.warning("Metric failed: %s", metric.__name__)
            return "N/A"

    @classmethod
    def analyze(cls, text: str) -> Dict[str, str]:
        logger.info("Running readability metrics")
        return {
            name: cls.apply_metric(metric, text) for name, metric in cls.METRICS.items()
        }


def split_chunks(content: str, chunk_size: int) -> Tuple[List[str], List[int]]:
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
