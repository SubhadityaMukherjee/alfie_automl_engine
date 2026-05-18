"""Integration tests for the AutoML+ router."""

import io
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.automlplus.router import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


# ---------------------------------------------------------------------------
# accepted_format
# ---------------------------------------------------------------------------


@patch(
    "app.automlplus.router.automl_plus_data_instructions", return_value="instructions"
)
def test_accepted_format(mock_instructions):
    resp = client.post("/automlplus/accepted_format/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


# ---------------------------------------------------------------------------
# image_to_website
# ---------------------------------------------------------------------------


def test_image_to_website_not_implemented():
    resp = client.post("/automlplus/image_tools/image_to_website/")
    assert resp.status_code == 501


# ---------------------------------------------------------------------------
# check-alt-text
# ---------------------------------------------------------------------------


@patch("app.automlplus.router.AltTextChecker")
def test_check_alt_text_success(mock_checker_cls):
    mock_checker_cls.check.return_value = "Good alt text"
    resp = client.post(
        "/automlplus/web_access/check-alt-text/",
        data={"image_url": "http://img.png", "alt_text": "desc"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["src"] == "http://img.png"
    assert body["alt_text"] == "desc"


@patch("app.automlplus.router.AltTextChecker")
def test_check_alt_text_error(mock_checker_cls):
    mock_checker_cls.check.side_effect = RuntimeError("VLM error")
    resp = client.post(
        "/automlplus/web_access/check-alt-text/",
        data={"image_url": "http://img.png", "alt_text": "desc"},
    )
    assert resp.status_code == 500
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# run_on_image
# ---------------------------------------------------------------------------


def test_run_on_image_missing_image():
    resp = client.post(
        "/automlplus/image_tools/run_on_image/",
        data={"prompt": "describe"},
    )
    assert resp.status_code == 400


@patch("app.automlplus.router.ImagePromptRunner")
def test_run_on_image_success(mock_runner):
    mock_runner.run.return_value = "A cat"
    resp = client.post(
        "/automlplus/image_tools/run_on_image/",
        data={"prompt": "describe"},
        files={"image_file": ("test.png", b"fake-image", "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "A cat"


@patch("app.automlplus.router.ImagePromptRunner")
def test_run_on_image_error(mock_runner):
    mock_runner.run.side_effect = RuntimeError("fail")
    resp = client.post(
        "/automlplus/image_tools/run_on_image/",
        data={"prompt": "describe"},
        files={"image_file": ("test.png", b"fake-image", "image/png")},
    )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# run_on_image_stream
# ---------------------------------------------------------------------------


def test_run_on_image_stream_missing_image():
    resp = client.post(
        "/automlplus/image_tools/run_on_image_stream/",
        data={"prompt": "describe"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# web_access/analyze
# ---------------------------------------------------------------------------


@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_success(mock_extract, mock_analyzer, mock_pipeline):
    from app.automlplus.tools.text import ChunkResult

    mock_pipeline.return_value = [
        ChunkResult(
            chunk=0,
            start_line=1,
            end_line=10,
            score=80.0,
            image_feedback=[],
            llm_response="ok",
        )
    ]
    mock_analyzer.analyze.return_value = {"flesch_reading_ease": 80.0}

    html_file = io.BytesIO(b"<html><body>Hello</body></html>")
    resp = client.post(
        "/automlplus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["average_score"] == 80.0
    assert body["readability"]["flesch_reading_ease"] == 80.0


def test_analyze_missing_content():
    empty_file = io.BytesIO(b"")
    resp = client.post(
        "/automlplus/web_access/analyze/",
        files={"file": ("empty.html", empty_file, "text/html")},
    )
    assert resp.status_code == 400
