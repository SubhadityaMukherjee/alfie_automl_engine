import logging
import os
from typing import List

from jinja2 import Environment
from ollama import Client

from app.automlplus.utils import ImageConverter
from app.core.utils import render_template

logger = logging.getLogger(__name__)

client = Client()


class ImagePromptRunner:
    """Run a VLM on an image and user-provided prompt."""

    DEFAULT_MODEL = os.getenv("IMAGE_PROMPT_MODEL", "qwen2.5vl")

    @staticmethod
    def _resolve_model(model: str | None) -> str:
        if not model or not str(model).strip():
            return ImagePromptRunner.DEFAULT_MODEL
        return model

    @staticmethod
    def build_messages(
        jinja_environment: Environment | None, image_b64: str, prompt: str
    ) -> List[dict]:
        # If a jinja environment is provided, try to render a default system prompt; otherwise minimal messages
        messages: List[dict] = []
        if jinja_environment is not None:
            try:
                system_prompt = render_template(
                    jinja_environment, "image_to_website_prompt.txt"
                )
                messages.append({"role": "system", "content": system_prompt})
            except Exception:
                pass
        messages.append({"role": "user", "content": prompt, "images": [image_b64]})
        return messages

    @staticmethod
    def run(
        image_bytes: bytes | None = None,
        image_path_or_url: str | None = None,
        prompt: str = "",
        model: str | None = None,
        jinja_environment: Environment | None = None,
    ) -> str:
        model_name = ImagePromptRunner._resolve_model(model)
        try:
            if image_bytes is None and not image_path_or_url:
                raise ValueError("Provide either image_bytes or image_path_or_url")

            image_b64 = (
                ImageConverter.bytes_to_base64(image_bytes)
                if image_bytes is not None
                else ImageConverter.to_base64(str(image_path_or_url))
            )

            messages = ImagePromptRunner.build_messages(
                jinja_environment, image_b64, prompt
            )

            # Use low-level client to mirror ChatHandler.messages pattern
            result = client.chat(model=model_name, messages=messages)
            return result.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.exception("ImagePromptRunner failed")
            raise e
