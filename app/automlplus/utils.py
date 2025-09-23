
from io import BytesIO
import base64
import logging
import json

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


