"""Tests for app.automlplus.website_accessibility.crawler."""

from unittest.mock import MagicMock, patch

import pytest

from app.automlplus.website_accessibility.crawler import crawl_website
from app.core.exceptions import AutoMLValidationError

SEED = "https://example.com/"


def _mock_response(html: str) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


def _urls(result) -> list[str]:
    return [page.url for page in result.pages]


def test_invalid_depth_raises():
    with pytest.raises(AutoMLValidationError, match="depth must be >= 1"):
        crawl_website(SEED, depth=0)


def test_invalid_max_pages_raises():
    with pytest.raises(AutoMLValidationError, match="max_pages must be >= 1"):
        crawl_website(SEED, depth=1, max_pages=0)


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_depth_one_fetches_only_seed(mock_get):
    mock_get.return_value = _mock_response(
        '<a href="/about">About</a><a href="/contact">Contact</a>'
    )
    result = crawl_website(SEED, depth=1)
    assert _urls(result) == ["https://example.com"]
    assert mock_get.call_count == 1


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_depth_two_follows_same_origin_links(mock_get):
    def get(url, **kwargs):
        if url.rstrip("/") == "https://example.com":
            return _mock_response('<a href="/about">About</a>')
        return _mock_response("<p>sub page</p>")

    mock_get.side_effect = get
    result = crawl_website(SEED, depth=2)
    assert _urls(result) == ["https://example.com", "https://example.com/about"]


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_depth_three_follows_two_levels(mock_get):
    def get(url, **kwargs):
        if url.rstrip("/") == "https://example.com":
            return _mock_response('<a href="/level1">L1</a>')
        if url == "https://example.com/level1":
            return _mock_response('<a href="/level2">L2</a>')
        return _mock_response("<p>leaf</p>")

    mock_get.side_effect = get
    result = crawl_website(SEED, depth=3)
    assert _urls(result) == [
        "https://example.com",
        "https://example.com/level1",
        "https://example.com/level2",
    ]


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_external_links_are_skipped(mock_get):
    mock_get.return_value = _mock_response(
        '<a href="https://other.com/page">External</a>'
    )
    result = crawl_website(SEED, depth=2)
    assert _urls(result) == ["https://example.com"]


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_non_html_assets_are_skipped(mock_get):
    mock_get.return_value = _mock_response(
        '<a href="/img/photo.jpg">pic</a>'
        '<a href="/doc.pdf">doc</a>'
        '<a href="/style.css">css</a>'
        '<a href="mailto:a@b.com">mail</a>'
    )
    result = crawl_website(SEED, depth=2)
    assert _urls(result) == ["https://example.com"]


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_duplicate_and_fragment_links_deduped(mock_get):
    mock_get.return_value = _mock_response(
        '<a href="/about">1</a><a href="/about/">2</a><a href="/about#sec">3</a>'
    )
    result = crawl_website(SEED, depth=2)
    assert _urls(result) == ["https://example.com", "https://example.com/about"]


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_max_pages_caps_crawl(mock_get):
    mock_get.return_value = _mock_response(
        '<a href="/a">a</a><a href="/b">b</a><a href="/c">c</a>'
    )
    result = crawl_website(SEED, depth=2, max_pages=2)
    assert len(result.pages) == 2


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_seed_failure_raises(mock_get):
    mock_get.side_effect = Exception("boom")
    with pytest.raises(Exception, match="boom"):
        crawl_website(SEED, depth=1)


@patch("app.automlplus.website_accessibility.crawler.requests.get")
def test_subpage_failure_is_collected_not_raised(mock_get):
    def get(url, **kwargs):
        if url.rstrip("/") == "https://example.com":
            return _mock_response('<a href="/broken">broken</a><a href="/ok">ok</a>')
        if url.endswith("/broken"):
            raise Exception("404")
        return _mock_response("<p>ok</p>")

    mock_get.side_effect = get
    result = crawl_website(SEED, depth=2)
    assert _urls(result) == ["https://example.com", "https://example.com/ok"]
    assert result.errors == [{"url": "https://example.com/broken", "error": "404"}]
