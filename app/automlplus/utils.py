import base64
import logging
import os
from io import BytesIO
from typing import Any

import requests
from bs4 import BeautifulSoup
from PIL import Image

logger = logging.getLogger(__name__)


class ImageConverter:
    """Convert images to base64 from local paths or URLs."""

    @staticmethod
    def to_base64(image_path_or_url: str) -> str:
        logger.info("Converting image to base64: %s", image_path_or_url)
        try:
            if image_path_or_url.startswith("http"):
                headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageConverter/1.0)"}
                resp = requests.get(image_path_or_url, headers=headers)
                resp.raise_for_status()
                if "image" not in resp.headers.get("Content-Type", ""):
                    raise ValueError(
                        f"URL does not point to an image: {image_path_or_url}"
                    )
                image = Image.open(BytesIO(resp.content))
            else:
                if not os.path.isfile(image_path_or_url):
                    raise FileNotFoundError(f"No such file: {image_path_or_url}")
                image = Image.open(image_path_or_url)

            image = image.convert("RGBA")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.exception("Image conversion failed")
            raise e

    @staticmethod
    def bytes_to_base64(image_bytes: bytes) -> str:
        """Convert raw image bytes to base64 PNG string."""
        try:
            image = Image.open(BytesIO(image_bytes))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.exception("Image bytes conversion failed")
            raise e


def extract_text_from_html_bytes(content: bytes) -> str:
    """Extract readable text from raw HTML bytes."""
    soup = BeautifulSoup(content, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    lines = (line.strip() for line in soup.get_text().splitlines())
    phrases = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in phrases if chunk)
    return text


def json_safe(data: Any) -> Any:
    """Recursively convert string values to JSON-safe strings."""
    if isinstance(data, dict):
        return {k: json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_safe(v) for v in data]
    elif isinstance(data, str):
        return (
            data.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
    else:
        return data
