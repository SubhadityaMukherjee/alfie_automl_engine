import asyncio
import json
from types import SimpleNamespace

import pytest

from app.website_accessibility.services import (
    extract_text_from_html_bytes,
    run_accessibility_pipeline,
    stream_accessibility_results,
)


def test_extract_text_from_html_bytes_basic():
    html = b"""
    <html><head><style>.x{}</style><script>var a=1;</script></head>
    <body>
      <h1>Title</h1>
      <p>Para one.</p>
      <p>Para two.</p>
    </body>
    </html>
    """
    text = extract_text_from_html_bytes(html)
    # Ensure script/style are removed and text preserved
    assert "Title" in text
    assert "Para one." in text
    assert "Para two." in text
    assert "var a=1" not in text


@pytest.mark.asyncio
async def test_run_pipeline_and_stream(monkeypatch):
    # Mock ChatHandler.chat to return a deterministic response containing a Score
    from app.core import chat_handler

    async def fake_chat(prompt: str, context: str, stream: bool):
        return "Score: 7.5\nOK"

    monkeypatch.setattr(chat_handler.ChatHandler, "chat", staticmethod(fake_chat))

    # Mock AltTextChecker to avoid real image checks
    from app.website_accessibility import modules as wa_modules

    def fake_check(env, src, alt):
        return "Looks good"

    monkeypatch.setattr(wa_modules.AltTextChecker, "check", staticmethod(fake_check))

    content = """
    <html>
      <body>
        <img src=\"x.png\" alt=\"desc\" />
        <p>Lorem ipsum dolor sit amet.</p>
      </body>
    </html>
    """

    # Minimal jinja environment stub
    jinja_env = SimpleNamespace()

    results = await run_accessibility_pipeline(
        content=content,
        filename="test.html",
        jinja_environment=jinja_env,
        chunk_size=1000,
        concurrency=2,
    )

    assert isinstance(results, list)
    assert len(results) >= 1
    first = results[0]
    assert hasattr(first, "chunk")
    assert hasattr(first, "start_line")
    assert hasattr(first, "end_line")
    assert first.score is None or isinstance(first.score, float)

    # Test streaming helper
    chunks = []
    async for b in stream_accessibility_results(results):
        chunks.append(b.decode("utf-8").strip())
    assert len(chunks) >= 1
    summary = json.loads(chunks[-1])
    assert "average_score" in summary


