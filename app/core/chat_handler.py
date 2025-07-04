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
