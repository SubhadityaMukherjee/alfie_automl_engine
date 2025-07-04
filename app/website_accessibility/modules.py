import base64
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import List

import requests
import textstat
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader
from ollama import Client
from PIL import Image
from utils import render_template
import ollama
import asyncio
import logging

logger = logging.getLogger(__name__)

client = Client()
jinja_path = os.getenv("JINJAPATH") or ""
jinja_environment = Environment(loader=FileSystemLoader(Path(jinja_path)))


class ImageConverter:
    @staticmethod
    def to_base64(image_path_or_url: str):
        logger.info("Converting image to base64: %s", image_path_or_url)
        try:
            if image_path_or_url.startswith("http"):
                image = Image.open(requests.get(image_path_or_url, stream=True).raw)
            else:
                image = Image.open(image_path_or_url)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.exception("Image conversion failed")
            raise e


class AltTextChecker:
    @staticmethod
    def check(jinja_environment, image_url_or_path, alt_text, model="qwen2.5vl"):
        logger.info("Checking alt-text using model %s", model)
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
            response = client.chat(model=model, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            logger.exception("AltTextChecker failed")
            raise e


class ReadabilityAnalyzer:
    METRICS = {
        "Flesch Reading Ease": textstat.flesch_reading_ease,
        "Difficult Words": textstat.difficult_words,
        "Lexicon Count": textstat.lexicon_count,
        "Avg Sentence Length": textstat.avg_sentence_length,
    }

    @staticmethod
    def apply_metric(metric, text):
        try:
            return metric(text)
        except Exception:
            logger.warning("Metric failed: %s", metric.__name__)
            return "N/A"

    @classmethod
    def analyze(cls, text):
        logger.info("Running readability metrics")
        return {
            name: cls.apply_metric(metric, text) for name, metric in cls.METRICS.items()
        }


def split_chunks(content, chunk_size):
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


class ChatQueue:
    def __init__(self, num_workers=4):
        self.queue = asyncio.Queue()
        self.tasks = []
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)

    async def start(self):
        self.tasks = [
            asyncio.create_task(self.worker()) for _ in range(self.num_workers)
        ]

    async def worker(self):
        while True:
            fut, message, context, model, stream, stream_queue = await self.queue.get()
            try:
                if stream:
                    async for chunk in ChatHandler._chat_stream_internal(
                        message, context, model
                    ):
                        await stream_queue.put(chunk)
                    await stream_queue.put(None)
                    fut.set_result(True)
                else:
                    result = await ChatHandler._chat_internal(message, context, model)
                    fut.set_result(result)
            except Exception as e:
                if stream:
                    await stream_queue.put(f"[ERROR] {str(e)}")
                    await stream_queue.put(None)
                fut.set_exception(e)
            finally:
                self.queue.task_done()

    async def submit(self, message, context="", model="gemma3:4b", stream=False):
        async with self.semaphore:
            if stream:

                async def stream_gen():
                    async for chunk in ChatHandler._chat_stream_internal(
                        message, context, model
                    ):
                        yield chunk

                return stream_gen()
            else:
                return await ChatHandler._chat_internal(message, context, model)


class ChatHandler:
    queue = ChatQueue()

    @staticmethod
    async def init():
        await ChatHandler.queue.start()

    @staticmethod
    async def chat(
        message, context: str = "", model: str = "gemma3:4b", stream: bool = False
    ):
        return await ChatHandler.queue.submit(message, context, model, stream)

    @staticmethod
    async def _chat_internal(message, context: str, model: str):
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": message},
                {"role": "user-hidden", "content": context},
            ],
        )
        return response["message"]["content"].strip()

    @staticmethod
    async def _chat_stream_internal(message, context: str, model: str):
        import ollama

        stream = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": message},
                {"role": "user-hidden", "content": context},
            ],
            stream=True,
        )

        async for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
