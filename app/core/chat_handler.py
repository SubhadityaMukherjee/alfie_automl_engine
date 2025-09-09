"""Chat handling utilities for async LLM requests.

Provides a simple queued interface (`ChatQueue`) and a static facade
(`ChatHandler`) to interact with local Ollama models, supporting both
regular and streaming responses.
"""
import asyncio
import base64
import logging
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import List

import ollama
import requests
import textstat
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader
from ollama import Client
from PIL import Image

from app.core.utils import render_template

# Load environment variables from the project root .env
load_dotenv(find_dotenv())


class ChatQueue:
    """A lightweight async work queue for LLM chat requests.

    Tasks submitted here are processed by background workers to avoid
    blocking the main event loop. Supports both streaming and non-streaming
    chat calls.

    Parameters
    ----------
    num_workers: int
        Number of concurrent workers processing chat requests.
    """

    def __init__(self, num_workers=4):
        self.queue = asyncio.Queue()
        self.tasks = []
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)

    async def start(self):
        """Start background worker tasks."""
        self.tasks = [
            asyncio.create_task(self.worker()) for _ in range(self.num_workers)
        ]

    async def worker(self):
        """Continuously pull requests off the queue and process them."""
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
        """Submit a chat request.

        Parameters
        ----------
        message: str
            The user message to send to the model.
        context: str
            Optional hidden context to include for the model.
        model: str
            Ollama model name to use.
        stream: bool
            If True, returns an async generator yielding chunks.
        """
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
        """Initialize the underlying queue workers."""
        await ChatHandler.queue.start()

    @staticmethod
    async def chat(
        message, context: str = "", model: str = "gemma3:4b", stream: bool = False
    ):
        """Public API to perform a chat call.

        When `stream` is True, returns an async generator of text chunks,
        otherwise returns the full response string.
        """
        return await ChatHandler.queue.submit(message, context, model, stream)

    @staticmethod
    async def _chat_internal(message, context: str, model: str):
        """Execute a single, non-streaming chat call against Ollama."""
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
        """Execute a streaming chat call against Ollama, yielding content chunks."""
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
