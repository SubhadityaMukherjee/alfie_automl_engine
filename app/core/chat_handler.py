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
    """Async work queue for chat requests, supporting streaming and non-streaming calls."""

    def __init__(self, num_workers=4):
        self.queue = asyncio.Queue()
        self.tasks = []
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)

    async def start(self):
        self.tasks = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]

    async def worker(self):
        while True:
            fut, message, context, backend, model, stream, stream_queue = await self.queue.get()
            try:
                if stream:
                    async for chunk in ChatHandler.dispatch_stream(message, context, backend, model):
                        await stream_queue.put(chunk)
                    await stream_queue.put(None)
                    fut.set_result(True)
                else:
                    result = await ChatHandler.dispatch(message, context, backend, model)
                    fut.set_result(result)
            except Exception as e:
                if stream:
                    await stream_queue.put(f"[ERROR] {str(e)}")
                    await stream_queue.put(None)
                fut.set_exception(e)
            finally:
                self.queue.task_done()

    async def submit(self, message, context="", backend="ollama", model="gemma3:4b", stream=False):
        async with self.semaphore:
            if stream:

                async def stream_gen():
                    async for chunk in ChatHandler.dispatch_stream(message, context, backend, model):
                        yield chunk

                return stream_gen()
            else:
                return await ChatHandler.dispatch(message, context, backend, model)


class ChatHandler:
    queue = ChatQueue()

    @staticmethod
    async def init():
        await ChatHandler.queue.start()

    @staticmethod
    async def chat(message, context="", backend="ollama", model="gemma3:4b", stream=False):
        return await ChatHandler.queue.submit(message, context, backend, model, stream)

    @staticmethod
    async def dispatch(message, context, backend, model):
        """Route chat requests to the correct backend."""
        if backend.lower() == "ollama":
            return await ChatHandler._ollama_chat(message, context, model)
        elif backend.lower() == "azure":
            return await ChatHandler._azure_chat(message, context, model)
        else:
            raise ValueError(f"Unknown chat backend: {backend}")

    @staticmethod
    async def dispatch_stream(message, context, backend, model):
        if backend.lower() == "ollama":
            async for chunk in ChatHandler._ollama_chat_stream(message, context, model):
                yield chunk
        elif backend.lower() == "azure":
            async for chunk in ChatHandler._azure_chat_stream(message, context, model):
                yield chunk
        else:
            raise ValueError(f"Unknown chat backend: {backend}")

    # --- Ollama implementation ---
    @staticmethod
    async def _ollama_chat(message, context, model):
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
    async def _ollama_chat_stream(message, context, model):
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

    # --- Azure placeholders ---
    @staticmethod
    async def _azure_chat(message, context, model):
        # TODO: implement Azure chat
        return "Azure chat not implemented yet."

    @staticmethod
    async def _azure_chat_stream(message, context, model):
        # TODO: implement Azure streaming chat
        if False:
            yield

