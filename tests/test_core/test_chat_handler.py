"""Tests for app.core.chat_handler.ChatHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.chat_handler import _MAX_CONCURRENT, ChatHandler

# ---------------------------------------------------------------------------
# Semaphore
# ---------------------------------------------------------------------------


def test_semaphore_limit():
    assert ChatHandler._semaphore._value == _MAX_CONCURRENT
    assert _MAX_CONCURRENT == 4


# ---------------------------------------------------------------------------
# _get_azure_client
# ---------------------------------------------------------------------------


def test_get_azure_client_missing_env_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(
            RuntimeError, match="Missing AZURE_OPENAI_ENDPOINT_LARGE_MODEL"
        ):
            ChatHandler._get_azure_client()


@patch("app.core.chat_handler.ChatCompletionsClient")
@patch("app.core.chat_handler.AzureKeyCredential")
def test_get_azure_client_success(mock_cred, mock_client):
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_ENDPOINT_LARGE_MODEL": "https://example.com",
            "AZURE_OPENAI_KEY": "key123",
        },
    ):
        client = ChatHandler._get_azure_client()
        mock_client.assert_called_once()
        assert client == mock_client.return_value


@patch("app.core.chat_handler.ChatCompletionsClient", side_effect=Exception("boom"))
def test_get_azure_client_init_failure(mock_client_cls):
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_ENDPOINT_LARGE_MODEL": "https://example.com",
            "AZURE_OPENAI_KEY": "key123",
        },
    ):
        with pytest.raises(RuntimeError, match="Failed to initialize Azure client"):
            ChatHandler._get_azure_client()


# ---------------------------------------------------------------------------
# _extract_azure_text_from_response
# ---------------------------------------------------------------------------


def test_extract_text_choices_str():
    msg = MagicMock()
    msg.content = "  hello world  "
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    assert ChatHandler._extract_azure_text_from_response(response) == "hello world"


def test_extract_text_choices_list():
    part1 = MagicMock()
    part1.text = "hello "
    part2 = {"text": "world"}
    part3 = "!"
    msg = MagicMock()
    msg.content = [part1, part2, part3]
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    assert ChatHandler._extract_azure_text_from_response(response) == "hello world!"


def test_extract_text_choices_dict_message():
    msg = {"content": "dict content"}
    choice = {"message": msg}
    response = MagicMock()
    response.choices = [choice]
    assert ChatHandler._extract_azure_text_from_response(response) == "dict content"


def test_extract_text_output_message_fallback():
    content_item = MagicMock()
    content_item.text = "  fallback text  "
    output_msg = MagicMock()
    output_msg.content = [content_item]
    response = MagicMock()
    response.choices = []
    response.output_message = output_msg
    assert ChatHandler._extract_azure_text_from_response(response) == "fallback text"


def test_extract_text_str_fallback():
    response = "plain string response"
    assert (
        ChatHandler._extract_azure_text_from_response(response)
        == "plain string response"
    )


# ---------------------------------------------------------------------------
# _to_azure_messages
# ---------------------------------------------------------------------------


def test_to_azure_messages_system():
    from azure.ai.inference.models import SystemMessage

    msgs = [{"role": "system", "content": "You are helpful"}]
    result = ChatHandler._to_azure_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)


def test_to_azure_messages_user():
    from azure.ai.inference.models import UserMessage

    msgs = [{"role": "user", "content": "Hello"}]
    result = ChatHandler._to_azure_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)


def test_to_azure_messages_with_images():
    msgs = [{"role": "user", "content": "Describe", "images": ["base64data123"]}]
    result = ChatHandler._to_azure_messages(msgs)
    assert len(result) == 1
    content = result[0].content
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe"}
    assert content[1]["type"] == "image_url"


def test_to_azure_messages_images_downgrade_system_to_user():
    from azure.ai.inference.models import UserMessage

    msgs = [{"role": "system", "content": "sys", "images": ["base64data"]}]
    result = ChatHandler._to_azure_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)


def test_to_azure_messages_empty_images_skipped():
    msgs = [{"role": "user", "content": "Hi", "images": [None, ""]}]
    result = ChatHandler._to_azure_messages(msgs)
    assert len(result) == 1


def test_to_azure_messages_no_images_uses_text_only():
    from azure.ai.inference.models import UserMessage

    msgs = [{"role": "user", "content": "Hello"}]
    result = ChatHandler._to_azure_messages(msgs)
    assert isinstance(result[0], UserMessage)
    assert result[0].content == "Hello"


# ---------------------------------------------------------------------------
# dispatch / dispatch_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown chat backend"):
        await ChatHandler.dispatch("msg", "", "unknown_backend", "model")


@pytest.mark.asyncio
@patch.object(ChatHandler, "_azure_chat", return_value="response text")
async def test_dispatch_azure_calls_azure_chat(mock_azure):
    result = await ChatHandler.dispatch("msg", "ctx", "azure", "gpt-4o-mini")
    mock_azure.assert_called_once_with("msg", "ctx", "gpt-4o-mini")
    assert result == "response text"


@pytest.mark.asyncio
async def test_dispatch_stream_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown chat backend"):
        chunks = []
        async for c in ChatHandler.dispatch_stream("msg", "", "unknown", "model"):
            chunks.append(c)


# ---------------------------------------------------------------------------
# chat (facade)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch.object(ChatHandler, "dispatch", return_value="hi")
async def test_chat_non_streaming_returns_string(mock_dispatch):
    result = await ChatHandler.chat("msg", context="", stream=False)
    assert result == "hi"


@pytest.mark.asyncio
@patch.object(ChatHandler, "dispatch_stream")
async def test_chat_streaming_returns_async_generator(mock_dispatch_stream):
    async def fake_gen():
        yield "chunk1"
        yield "chunk2"

    mock_dispatch_stream.return_value = fake_gen()
    result = await ChatHandler.chat("msg", context="", stream=True)
    chunks = []
    async for c in result:
        chunks.append(c)
    assert chunks == ["chunk1", "chunk2"]


# ---------------------------------------------------------------------------
# _azure_chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch.object(ChatHandler, "_get_azure_client")
@patch.object(
    ChatHandler, "_extract_azure_text_from_response", return_value="extracted"
)
async def test_azure_chat_returns_extracted_text(mock_extract, mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.complete.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = await ChatHandler._azure_chat("msg", "ctx", "gpt-4o-mini")
    assert result == "extracted"


@pytest.mark.asyncio
@patch.object(ChatHandler, "_get_azure_client")
async def test_azure_chat_exception_propagates(mock_get_client):
    mock_client = MagicMock()
    mock_client.complete.side_effect = Exception("network error")
    mock_get_client.return_value = mock_client

    with pytest.raises(Exception, match="network error"):
        await ChatHandler._azure_chat("msg", "ctx", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# _azure_chat_stream
# ---------------------------------------------------------------------------


def _make_mock_stream(events):
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter(events))
    stream.close = MagicMock()
    return stream


@pytest.mark.asyncio
@patch("app.core.chat_handler.asyncio.get_running_loop")
@patch.object(ChatHandler, "_get_azure_client")
async def test_azure_chat_stream_yields_str_content(mock_get_client, mock_get_loop):
    delta = MagicMock()
    delta.content = "hello"
    event = MagicMock()
    event.delta = delta
    mock_stream = _make_mock_stream([event])

    mock_client = MagicMock()
    mock_client.complete.return_value = mock_stream
    mock_get_client.return_value = mock_client

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=mock_stream)
    mock_get_loop.return_value = mock_loop

    chunks = []
    async for c in ChatHandler._azure_chat_stream("msg", "ctx", "gpt-4o-mini"):
        chunks.append(c)
    assert chunks == ["hello"]


@pytest.mark.asyncio
@patch("app.core.chat_handler.asyncio.get_running_loop")
@patch.object(ChatHandler, "_get_azure_client")
async def test_azure_chat_stream_yields_dict_content(mock_get_client, mock_get_loop):
    delta = {"content": "dict_text"}
    event = MagicMock()
    event.delta = delta
    mock_stream = _make_mock_stream([event])

    mock_client = MagicMock()
    mock_client.complete.return_value = mock_stream
    mock_get_client.return_value = mock_client

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=mock_stream)
    mock_get_loop.return_value = mock_loop

    chunks = []
    async for c in ChatHandler._azure_chat_stream("msg", "ctx", "gpt-4o-mini"):
        chunks.append(c)
    assert chunks == ["dict_text"]


@pytest.mark.asyncio
@patch("app.core.chat_handler.asyncio.get_running_loop")
@patch.object(ChatHandler, "_get_azure_client")
async def test_azure_chat_stream_yields_list_content(mock_get_client, mock_get_loop):
    item1 = MagicMock()
    item1.text = "part1"
    item2 = {"text": "part2"}
    delta = MagicMock()
    delta.content = [item1, item2, "part3"]
    event = MagicMock()
    event.delta = delta
    mock_stream = _make_mock_stream([event])

    mock_client = MagicMock()
    mock_client.complete.return_value = mock_stream
    mock_get_client.return_value = mock_client

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=mock_stream)
    mock_get_loop.return_value = mock_loop

    chunks = []
    async for c in ChatHandler._azure_chat_stream("msg", "ctx", "gpt-4o-mini"):
        chunks.append(c)
    assert chunks == ["part1", "part2", "part3"]


@pytest.mark.asyncio
@patch("app.core.chat_handler.asyncio.get_running_loop")
@patch.object(ChatHandler, "_get_azure_client")
async def test_azure_chat_stream_closes_stream(mock_get_client, mock_get_loop):
    event = MagicMock()
    event.delta = MagicMock()
    event.delta.content = None
    mock_stream = _make_mock_stream([event])

    mock_client = MagicMock()
    mock_client.complete.return_value = mock_stream
    mock_get_client.return_value = mock_client

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=mock_stream)
    mock_get_loop.return_value = mock_loop

    chunks = []
    async for c in ChatHandler._azure_chat_stream("msg", "ctx", "gpt-4o-mini"):
        chunks.append(c)
    mock_stream.close.assert_called_once()


@pytest.mark.asyncio
@patch("app.core.chat_handler.asyncio.get_running_loop")
@patch.object(ChatHandler, "_get_azure_client")
async def test_azure_chat_stream_exception_propagates(mock_get_client, mock_get_loop):
    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(side_effect=Exception("stream error"))
    mock_get_loop.return_value = mock_loop
    mock_get_client.return_value = MagicMock()

    with pytest.raises(Exception):
        async for _ in ChatHandler._azure_chat_stream("msg", "ctx", "gpt-4o-mini"):
            pass


# ---------------------------------------------------------------------------
# chat_sync_messages
# ---------------------------------------------------------------------------


@patch.object(ChatHandler, "_azure_chat_messages_sync", return_value="sync_result")
def test_chat_sync_messages_azure(mock_sync):
    result = ChatHandler.chat_sync_messages([{"role": "user", "content": "hi"}])
    assert result == "sync_result"


def test_chat_sync_messages_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown chat backend"):
        ChatHandler.chat_sync_messages(
            [{"role": "user", "content": "hi"}], backend="unknown"
        )


# ---------------------------------------------------------------------------
# chat_stream_messages_sync
# ---------------------------------------------------------------------------


def test_chat_stream_messages_sync_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown chat backend"):
        ChatHandler.chat_stream_messages_sync(
            [{"role": "user", "content": "hi"}], backend="unknown"
        )


# ---------------------------------------------------------------------------
# _azure_chat_messages_sync
# ---------------------------------------------------------------------------


@patch.object(ChatHandler, "_get_azure_client")
@patch.object(
    ChatHandler, "_extract_azure_text_from_response", return_value="msg_result"
)
def test_azure_chat_messages_sync(mock_extract, mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.complete.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = ChatHandler._azure_chat_messages_sync(
        [{"role": "user", "content": "hi"}], "gpt-4o-mini"
    )
    assert result == "msg_result"


# ---------------------------------------------------------------------------
# _azure_chat_messages_stream_sync
# ---------------------------------------------------------------------------


@patch.object(ChatHandler, "_get_azure_client")
def test_azure_chat_messages_stream_sync_yields(mock_get_client):
    delta = MagicMock()
    delta.content = "stream_chunk"
    event = MagicMock()
    event.delta = delta
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([event]))
    mock_stream.close = MagicMock()

    mock_client = MagicMock()
    mock_client.complete.return_value = mock_stream
    mock_get_client.return_value = mock_client

    chunks = list(
        ChatHandler._azure_chat_messages_stream_sync(
            [{"role": "user", "content": "hi"}], "gpt-4o-mini"
        )
    )
    assert chunks == ["stream_chunk"]
