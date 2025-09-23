import base64
import json
import logging
from io import BytesIO

import requests
from PIL import Image
from urllib3.response import HTTPResponse

logger = logging.getLogger(__name__)


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

    @staticmethod
    def bytes_to_base64(image_bytes: bytes) -> str:
        """Convert raw image bytes to base64 PNG string."""
        try:
            image = Image.open(BytesIO(image_bytes))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.exception("Image bytes conversion failed")
            raise e
