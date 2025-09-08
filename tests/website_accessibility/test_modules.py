from io import BytesIO
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader
from PIL import Image

from app.website_accessibility import modules as wa_modules
from app.website_accessibility.modules import ReadabilityAnalyzer, split_chunks


class TestReadabilityAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return ReadabilityAnalyzer()

    @pytest.fixture
    def analyzed_chunk(self, analyzer):
        return analyzer.analyze("Hello I have to be analyzed. I am scared")

    def test_metrics_present(self, analyzed_chunk):
        response = analyzed_chunk
        assert "Flesch Reading Ease" in response
        assert "Difficult Words" in response
        assert "Lexicon Count" in response
        assert "Avg Sentence Length" in response


class TestSplitChunks:
    def test_split_chunks(self):
        test_string = "hello I am a test string that will be split. Oh no"
        chunks, line_ranges = split_chunks(test_string, chunk_size=10)
        assert chunks[0] == "hello I am"
        assert chunks[-1] == "lit. Oh no"
        assert line_ranges[0] == (1, 1)
        assert line_ranges[-1] == (1, 1)


class TestImageConverter:
    def test_to_base64_from_local_path(self, tmp_path: Path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (2, 2), color=(255, 0, 0))
        img.save(img_path, format="PNG")

        encoded = wa_modules.ImageConverter.to_base64(str(img_path))
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_to_base64_from_url(self, monkeypatch):
        # Create a fake image stream
        img = Image.new("RGB", (2, 2), color=(0, 255, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        class FakeResponse:
            def __init__(self, raw):
                self.raw = raw

        def fake_get(url, stream=True):
            return FakeResponse(raw=BytesIO(buf.getvalue()))

        monkeypatch.setattr(wa_modules.requests, "get", fake_get)

        encoded = wa_modules.ImageConverter.to_base64("http://example.com/image.png")
        assert isinstance(encoded, str)
        assert len(encoded) > 0


class TestAltTextChecker:
    def _make_env(self) -> Environment:
        templates_dir = Path("app/core/prompt_templates")
        return Environment(loader=FileSystemLoader(templates_dir))

    def test_check_returns_text(self, monkeypatch):
        # Mock ImageConverter to avoid actual file/network
        monkeypatch.setattr(
            wa_modules.ImageConverter,
            "to_base64",
            lambda _: "ZmFrZV9pbWFnZV9iNjQ=",
        )

        captured = {}

        class FakeClient:
            def chat(self, model, messages):
                captured["model"] = model
                captured["messages"] = messages
                return {"message": {"content": "Looks good"}}

        monkeypatch.setattr(wa_modules, "client", FakeClient())

        env = self._make_env()
        result = wa_modules.AltTextChecker.check(
            jinja_environment=env,
            image_url_or_path="http://example.com/i.png",
            alt_text="a sample alt text",
            model="test-model",
        )

        assert result == "Looks good"
        assert captured["model"] == "test-model"
        assert isinstance(captured["messages"], list)
        assert len(captured["messages"]) >= 3
        assert any(
            "Alt text:" in m.get("content", "") for m in captured["messages"]
        )  # user message
        assert any(
            "images" in m for m in captured["messages"]
        )  # image-attached message

    def test_check_uses_default_model_when_empty(self, monkeypatch):
        monkeypatch.setattr(
            wa_modules.ImageConverter,
            "to_base64",
            lambda _: "ZmFrZV9pbWFnZV9iNjQ=",
        )

        used_model = {"name": None}

        class FakeClient:
            def chat(self, model, messages):
                used_model["name"] = model
                return {"message": {"content": "ok"}}

        monkeypatch.setattr(wa_modules, "client", FakeClient())

        env = self._make_env()
        _ = wa_modules.AltTextChecker.check(
            jinja_environment=env,
            image_url_or_path="/tmp/i.png",
            alt_text="desc",
            model="",
        )

        assert used_model["name"] == "qwen2.5vl"
