import base64
import logging
from io import BytesIO
import os

import requests
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
                    raise ValueError(f"URL does not point to an image: {image_path_or_url}")
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
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.exception("Image bytes conversion failed")
            raise e
