"""Integration tests for the AutoML+ router."""

import io
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router

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
    resp = client.post("/automl/automl_plus/accepted_format/")
    assert resp.status_code == 200
    assert "instructions" in resp.json()


# ---------------------------------------------------------------------------
# image_to_website
# ---------------------------------------------------------------------------


def test_image_to_website_not_implemented():
    resp = client.post("/automl/automl_plus/image_tools/image_to_website/")
    assert resp.status_code == 501


# ---------------------------------------------------------------------------
# check-alt-text
# ---------------------------------------------------------------------------


@patch("app.automlplus.router.AltTextChecker")
def test_check_alt_text_success(mock_checker_cls):
    mock_checker_cls.check.return_value = "Good alt text"
    resp = client.post(
        "/automl/automl_plus/web_access/check-alt-text/",
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
        "/automl/automl_plus/web_access/check-alt-text/",
        data={"image_url": "http://img.png", "alt_text": "desc"},
    )
    assert resp.status_code == 500
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# run_on_image
# ---------------------------------------------------------------------------


def test_run_on_image_missing_image():
    resp = client.post(
        "/automl/automl_plus/image_tools/run_on_image/",
        data={"prompt": "describe"},
    )
    assert resp.status_code == 400


@patch("app.automlplus.router.ImagePromptRunner")
def test_run_on_image_success(mock_runner):
    mock_runner.run.return_value = "A cat"
    resp = client.post(
        "/automl/automl_plus/image_tools/run_on_image/",
        data={"prompt": "describe"},
        files={"image_file": ("test.png", b"fake-image", "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "A cat"


@patch("app.automlplus.router.ImagePromptRunner")
def test_run_on_image_error(mock_runner):
    mock_runner.run.side_effect = RuntimeError("fail")
    resp = client.post(
        "/automl/automl_plus/image_tools/run_on_image/",
        data={"prompt": "describe"},
        files={"image_file": ("test.png", b"fake-image", "image/png")},
    )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# run_on_image_stream
# ---------------------------------------------------------------------------


def test_run_on_image_stream_missing_image():
    resp = client.post(
        "/automl/automl_plus/image_tools/run_on_image_stream/",
        data={"prompt": "describe"},
    )
    assert resp.status_code == 400


@patch("app.automlplus.router.ImagePromptRunner")
def test_run_on_image_stream_success(mock_runner):
    mock_runner.run_stream.return_value = iter(["chunk1", "chunk2"])
    resp = client.post(
        "/automl/automl_plus/image_tools/run_on_image_stream/",
        data={"prompt": "describe"},
        files={"image_file": ("test.png", b"fake-image", "image/png")},
    )
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# run_on_image with URL instead of file
# ---------------------------------------------------------------------------


@patch("app.automlplus.router.ImagePromptRunner")
def test_run_on_image_with_url(mock_runner):
    mock_runner.run.return_value = "A dog"
    resp = client.post(
        "/automl/automl_plus/image_tools/run_on_image/",
        data={"prompt": "describe", "image_url": "http://example.com/dog.png"},
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "A dog"


# ---------------------------------------------------------------------------
# web_access/analyze
# ---------------------------------------------------------------------------


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_success(mock_extract, mock_analyzer, mock_pipeline, mock_summary):
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
    mock_summary.return_value = "Summary report"

    html_file = io.BytesIO(b"<html><body>Hello</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["average_score"] == 80.0
    assert body["readability"]["flesch_reading_ease"] == 80.0
    assert body["summary"] == "Summary report"
    assert body["pages_crawled"] == ["test.html"]
    assert mock_pipeline.call_args.kwargs["page"] == "test.html"


def test_analyze_missing_inputs():
    resp = client.post("/automl/automl_plus/web_access/analyze/")
    assert resp.status_code == 400
    assert "Provide an HTML file or a url" in resp.json()["error"]


def test_analyze_invalid_depth():
    html_file = io.BytesIO(b"<html><body>Hello</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
        data={"depth": "0"},
    )
    assert resp.status_code == 400
    assert "depth must be >= 1" in resp.json()["error"]


def test_analyze_missing_content():
    empty_file = io.BytesIO(b"")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("empty.html", empty_file, "text/html")},
    )
    assert resp.status_code == 400


@patch("app.automlplus.router.crawl_website")
def test_analyze_url_fetch_error(mock_crawl):
    mock_crawl.side_effect = Exception("network error")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        data={"url": "http://example.com/bad.html"},
    )
    assert resp.status_code == 400
    assert "Failed to fetch URL" in resp.json()["error"]


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
@patch("app.automlplus.router.crawl_website")
def test_analyze_url_only_crawls_site(
    mock_crawl, mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
    from app.automlplus.tools.text import ChunkResult
    from app.automlplus.website_accessibility.crawler import CrawlResult, CrawledPage

    mock_crawl.return_value = CrawlResult(
        pages=[
            CrawledPage(url="http://example.com/", content="<html>home</html>"),
            CrawledPage(url="http://example.com/about", content="<html>about</html>"),
        ]
    )
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
    mock_summary.return_value = "Summary report"

    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        data={"url": "http://example.com/", "depth": "2"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "http://example.com/"
    assert body["pages_crawled"] == [
        "http://example.com/",
        "http://example.com/about",
    ]
    # Pipeline called once per crawled page
    assert mock_pipeline.call_count == 2
    assert mock_crawl.call_args.kwargs["depth"] == 2


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
@patch("app.automlplus.router.crawl_website")
def test_analyze_url_includes_crawl_errors(
    mock_crawl, mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
    from app.automlplus.tools.text import ChunkResult
    from app.automlplus.website_accessibility.crawler import CrawlResult, CrawledPage

    mock_crawl.return_value = CrawlResult(
        pages=[CrawledPage(url="http://example.com/", content="<html>home</html>")],
        errors=[{"url": "http://example.com/x", "error": "404"}],
    )
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
    mock_summary.return_value = "Summary report"

    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        data={"url": "http://example.com/"},
    )
    assert resp.status_code == 200
    assert resp.json()["crawl_errors"] == [
        {"url": "http://example.com/x", "error": "404"}
    ]


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_with_extra_context_file(
    mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
    from app.automlplus.tools.text import ChunkResult

    mock_pipeline.return_value = [
        ChunkResult(
            chunk=0,
            start_line=1,
            end_line=10,
            score=90.0,
            image_feedback=[],
            llm_response="ok",
        )
    ]
    mock_analyzer.analyze.return_value = {"flesch_reading_ease": 90.0}

    html_file = io.BytesIO(b"<html><body>Test</body></html>")
    context_file = io.BytesIO(b"Follow these guidelines: ...")

    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={
            "file": ("test.html", html_file, "text/html"),
            "extra_file_input": ("guidelines.txt", context_file, "text/plain"),
        },
    )
    assert resp.status_code == 200
    # Verify the pipeline was called with context
    call_kwargs = mock_pipeline.call_args[1]
    assert "Accessibility guidelines" in call_kwargs["context"]


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_readability_error_returns_error_in_payload(
    mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
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
    mock_analyzer.analyze.side_effect = RuntimeError("readability crash")

    html_file = io.BytesIO(b"<html><body>Test</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
    )
    assert resp.status_code == 200
    assert "error" in resp.json()["readability"]


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="")
def test_analyze_empty_text_skips_readability(
    mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
    from app.automlplus.tools.text import ChunkResult

    mock_pipeline.return_value = [
        ChunkResult(
            chunk=0,
            start_line=1,
            end_line=10,
            score=70.0,
            image_feedback=[],
            llm_response="ok",
        )
    ]

    html_file = io.BytesIO(b"<html><body>Test</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
    )
    assert resp.status_code == 200
    assert resp.json()["readability"] is None
    mock_analyzer.analyze.assert_not_called()


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_multiple_chunks_average_score(
    mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
    from app.automlplus.tools.text import ChunkResult

    mock_pipeline.return_value = [
        ChunkResult(
            chunk=0,
            start_line=1,
            end_line=10,
            score=60.0,
            image_feedback=[],
            llm_response="ok",
        ),
        ChunkResult(
            chunk=1,
            start_line=11,
            end_line=20,
            score=80.0,
            image_feedback=[],
            llm_response="ok",
        ),
    ]
    mock_analyzer.analyze.return_value = {"flesch_reading_ease": 70.0}

    html_file = io.BytesIO(b"<html><body>Test</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
    )
    assert resp.status_code == 200
    assert resp.json()["average_score"] == 70.0


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_summary_failure_is_not_fatal(
    mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
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
    mock_summary.side_effect = RuntimeError("LLM down")

    html_file = io.BytesIO(b"<html><body>Test</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
    )
    assert resp.status_code == 200
    assert resp.json()["summary"] is None


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_includes_html_report_by_default(
    mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
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
    mock_summary.return_value = "Summary report"

    html_file = io.BytesIO(b"<html><body>Test</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
    )
    assert resp.status_code == 200
    html_report = resp.json()["html_report"]
    assert html_report.startswith("<!DOCTYPE html>")
    # Untouched by json_safe: real newlines survive the round trip
    assert "\n" in html_report
    assert "Accessibility report" in html_report


@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_include_html_false_omits_report(
    mock_extract, mock_analyzer, mock_pipeline, mock_summary
):
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
    mock_summary.return_value = "Summary report"

    html_file = io.BytesIO(b"<html><body>Test</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
        data={"include_html": "false"},
    )
    assert resp.status_code == 200
    assert resp.json()["html_report"] is None


@patch("app.automlplus.router.render_report_html", side_effect=RuntimeError("boom"))
@patch("app.automlplus.router.summarize_accessibility_results", new_callable=AsyncMock)
@patch("app.automlplus.router.run_accessibility_pipeline")
@patch("app.automlplus.router.ReadabilityAnalyzer")
@patch("app.automlplus.router.extract_text_from_html_bytes", return_value="text")
def test_analyze_html_render_failure_is_not_fatal(
    mock_extract, mock_analyzer, mock_pipeline, mock_summary, mock_render
):
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
    mock_summary.return_value = "Summary report"

    html_file = io.BytesIO(b"<html><body>Test</body></html>")
    resp = client.post(
        "/automl/automl_plus/web_access/analyze/",
        files={"file": ("test.html", html_file, "text/html")},
    )
    assert resp.status_code == 200
    assert resp.json()["html_report"] is None
    assert resp.json()["average_score"] == 80.0
