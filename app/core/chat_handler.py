"""Chat handling utilities for async LLM requests.

Provides a simple queued interface (`ChatQueue`) and a static facade
(`ChatHandler`) to interact with Azure AI Inference models, supporting both
regular and streaming responses.
"""

import asyncio
import logging
import os
from typing import List

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


class ChatQueue:
    """Async work queue for chat requests, supporting streaming and non-streaming calls."""

    def __init__(self, num_workers=4):
        self.queue = asyncio.Queue()
        self.tasks = []
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)

    async def start(self):
        logger.debug("Started Chat task")
        self.tasks = [
            asyncio.create_task(self.worker()) for _ in range(self.num_workers)
        ]

    async def worker(self):
        while True:
            fut, message, context, backend, model, stream, stream_queue = (
                await self.queue.get()
            )
            try:
                if stream:
                    async for chunk in ChatHandler.dispatch_stream(
                        message, context, backend, model
                    ):
                        await stream_queue.put(chunk)
                    await stream_queue.put(None)
                    fut.set_result(True)
                else:
                    result = await ChatHandler.dispatch(
                        message, context, backend, model
                    )
                    fut.set_result(result)
            except Exception as e:
                if stream:
                    await stream_queue.put(f"[ERROR] {str(e)}")
                    await stream_queue.put(None)
                fut.set_exception(e)
            finally:
                self.queue.task_done()

    async def submit(
        self, message, context="", backend="azure", model="gpt-4o-mini", stream=False
    ):
        async with self.semaphore:

            logger.debug(
                f"Submitted {message} message with backend {backend}, model {model}, stream {stream}"
            )
            if stream:

                async def stream_gen():
                    async for chunk in ChatHandler.dispatch_stream(
                        message, context, backend, model
                    ):
                        yield chunk

                return stream_gen()
            else:
                return await ChatHandler.dispatch(message, context, backend, model)


class ChatHandler:
    queue = ChatQueue()

    @staticmethod
    async def init():
        logger.debug("Started queue")
        await ChatHandler.queue.start()

    @staticmethod
    async def chat(
        message, context="", backend="azure", model="gpt-4o-mini", stream=False
    ):
        return await ChatHandler.queue.submit(message, context, backend, model, stream)

    @staticmethod
    async def dispatch(message, context, backend, model):
        logger.debug(f"Dispatch Chat to backend {backend} with {message}, {context}")
        """Route chat requests to the correct backend."""
        if backend.lower() == "azure":
            return await ChatHandler._azure_chat(message, context, model)
        else:
            raise ValueError(f"Unknown chat backend: {backend}")

    @staticmethod
    async def dispatch_stream(message, context, backend, model):
        logger.debug(
            f"Dispatch Chat Stream to backend {backend} with {message}, {context}"
        )
        if backend.lower() == "azure":
            async for chunk in ChatHandler._azure_chat_stream(message, context, model):
                yield chunk
        else:
            raise ValueError(f"Unknown chat backend: {backend}")

    # --- Synchronous helpers for structured message payloads (incl. images) ---
    @staticmethod
    def chat_sync_messages(
        messages: List[dict], backend: str = "azure", model: str = "gpt-4o-mini"
    ) -> str:
        """Synchronously send a list of chat messages (optionally with images) to Azure."""
        logger.debug(f"Dispatch synchronous Chat to backend {backend} with {messages}")
        if backend.lower() == "azure":
            return ChatHandler._azure_chat_messages_sync(messages, model)
        else:
            raise ValueError(f"Unknown chat backend: {backend}")

    # --- Streaming helpers for structured message payloads (incl. images) ---
    @staticmethod
    def chat_stream_messages_sync(
        messages: List[dict], backend: str = "azure", model: str = "gpt-4o-mini"
    ):
        """Synchronously stream a list of chat messages (optionally with images).

        Yields incremental text chunks from Azure as they arrive.
        """
        logger.debug("Stream chat messages synchronously")
        if backend.lower() == "azure":
            return ChatHandler._azure_chat_messages_stream_sync(messages, model)
        else:
            raise ValueError(f"Unknown chat backend: {backend}")

    @staticmethod
    def _azure_chat_messages_stream_sync(messages: List[dict], model: str):
        client = ChatHandler._get_azure_client()
        azure_msgs = ChatHandler._to_azure_messages(messages)
        logger.debug("Azure client stream in chunks")
        stream = client.complete(model=model, messages=azure_msgs, stream=True)
        for event in stream:
            if hasattr(event, "delta") and event.delta:
                delta = event.delta
                content = getattr(delta, "content", None)
                if content is None and isinstance(delta, dict):
                    content = delta.get("content")
                if isinstance(content, str):
                    if content:
                        yield content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            if item:
                                yield item
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if text:
                                yield text
                        else:
                            text = getattr(item, "text", None)
                            if text:
                                yield text
        if hasattr(stream, "close"):
            stream.close()

    @staticmethod
    def _azure_chat_messages_sync(messages: List[dict], model: str) -> str:
        """Synchronous Azure chat that supports text and images.

        Accepts our internal message dicts of the form:
          {"role": "system"|"user", "content": str | None, "images": [base64str, ...]?}

        Converts them into Azure AI Inference message objects. For messages that
        include images, constructs a single UserMessage with a mixed content list
        containing a text item (when provided) and one input_image item per image.
        """
        client = ChatHandler._get_azure_client()
        logger.debug("Azure client init stream")
        azure_msgs = ChatHandler._to_azure_messages(messages)
        logger.debug(f"Azure message dict {azure_msgs}")
        response = client.complete(model=model, messages=azure_msgs)
        return ChatHandler._extract_azure_text_from_response(response)

    @staticmethod
    def _to_azure_messages(msgs: List[dict]) -> List[object]:
        """Convert internal message dicts to Azure AI Inference message objects."""
        azure_messages: List[object] = []

        for m in msgs:
            role = (m.get("role") or "user").lower()
            text_content = m.get("content")
            images = m.get("images") or []

            if not images:
                if role == "system":
                    azure_messages.append(SystemMessage(content=text_content or ""))
                else:
                    azure_messages.append(UserMessage(content=text_content or ""))
                continue

            mixed_content: List[object] = []
            if text_content:
                mixed_content.append({"type": "text", "text": text_content})
            for b64 in images:
                if not isinstance(b64, str) or not b64:
                    continue
                mixed_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )

            # Azure doesn't accept images in system role; downgrade to user
            azure_messages.append(UserMessage(content=mixed_content))

        return azure_messages

    @staticmethod
    def _get_azure_client():
        """Initialize Azure AI Foundry chat client."""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_LARGE_MODEL")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        if not endpoint or not api_key:
            raise RuntimeError(
                "Missing AZURE_OPENAI_ENDPOINT_LARGE_MODEL or AZURE_OPENAI_KEY environment variables"
            )
        logger.debug("Endpoint and API Key Exsists")
        return ChatCompletionsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )

    @staticmethod
    async def _azure_chat(message, context, model):
        """Non-streaming chat call using Azure GPT-4o-mini."""
        client = ChatHandler._get_azure_client()
        messages = [
            SystemMessage(content=context or "You are a helpful assistant."),
            UserMessage(content=message),
        ]
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: client.complete(model=model, messages=messages)
        )
        logger.debug("Azure chat async works")
        return ChatHandler._extract_azure_text_from_response(response)

    @staticmethod
    async def _azure_chat_stream(message, context, model):
        """Streaming chat call using Azure GPT-4o-mini."""
        client = ChatHandler._get_azure_client()
        messages = [
            SystemMessage(content=context or "You are a helpful assistant."),
            UserMessage(content=message),
        ]

        loop = asyncio.get_event_loop()

        def sync_stream():
            return client.complete(
                model=model,
                messages=messages,
                stream=True,
            )

        # Run the sync generator in a thread and forward chunks asynchronously
        stream = await loop.run_in_executor(None, sync_stream)
        for event in stream:
            if hasattr(event, "delta") and event.delta:
                delta = event.delta
                content = getattr(delta, "content", None)
                if content is None and isinstance(delta, dict):
                    content = delta.get("content")
                # content may be str or list of items (with text)
                if isinstance(content, str):
                    if content:
                        yield content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            if item:
                                yield item
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if text:
                                yield text
                        else:
                            text = getattr(item, "text", None)
                            if text:
                                yield text
        logger.debug("Azure chat stream works")
        if hasattr(stream, "close"):
            stream.close()

    @staticmethod
    def _extract_azure_text_from_response(response) -> str:
        """Extract text content from Azure ChatCompletions response across SDK variants."""
        # Try choices-based response (common structure)
        try:
            choice0 = response.choices[0]
            message = getattr(choice0, "message", None) or choice0["message"]
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            # content can be str or list of content parts
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        txt = part.get("text")
                        if txt:
                            parts.append(txt)
                    else:
                        txt = getattr(part, "text", None)
                        if txt:
                            parts.append(txt)
                return "".join(parts).strip()
        except Exception:
            pass
        # Fallback older shape: output_message.content[0].text
        try:
            return response.output_message.content[0].text.strip()
        except Exception:
            pass
        # Last resort: str(response)
        return str(response)
