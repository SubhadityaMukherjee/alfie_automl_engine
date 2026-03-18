import logging
import math

import pytest
import textstat

from app.automlplus.website_accessibility.modules import (ReadabilityAnalyzer,
                                                          split_chunks)

logger = logging.getLogger(__name__)


@pytest.fixture
def content_example_1():
    return "This is a long string"


@pytest.fixture
def content_example_2():
    return "This is an even longer string so we can check the chunking"


def test_split_chunks_2(content_example_1: str, chunk_size: int = 2) -> None:
    chunks, line_ranges = split_chunks(content_example_1, chunk_size=chunk_size)
    assert chunks == ["Th", "is", " i", "s ", "a ", "lo", "ng", " s", "tr", "in", "g"]
    assert line_ranges == [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ]


def test_split_chunks_20(content_example_2: str, chunk_size: int = 20) -> None:
    chunks, line_ranges = split_chunks(content_example_2, chunk_size=chunk_size)
    assert chunks == [
        "This is an even long",
        "er string so we can ",
        "check the chunking",
    ]
    assert line_ranges == [(1, 1), (1, 1), (1, 1)]


def test_ReadabilityAnalyzer_analyze_mocked_metrics(monkeypatch):
    fake_metrics = {
        "test_10": lambda _: 10,
        "test_20": lambda _: 20,
    }

    monkeypatch.setattr(ReadabilityAnalyzer, "METRICS", fake_metrics)

    result = ReadabilityAnalyzer.analyze("this is a text")
    assert result == {"test_10": 10, "test_20": 20}


def test_ReadabilityAnalyzer_analyze_apply_metric_can_return_string():
    def fake_metric(_):
        return "42"

    result = ReadabilityAnalyzer.apply_metric(fake_metric, "this is a text")
    assert result == "42"


def test_ReadabilityAnalyzer_analyze_apply_metric_can_return_int():
    def fake_metric(_):
        return 42

    result = ReadabilityAnalyzer.apply_metric(fake_metric, "this is a text")
    assert result == 42


def test_ReadabilityAnalyzer_analyze_apply_metric_can_return_float():
    def fake_metric(_):
        return 42.0

    result = ReadabilityAnalyzer.apply_metric(fake_metric, "this is a text")
    assert result == 42.0


def test_ReadabilityAnalyzer_analyze_apply_metric_failed():
    def fake_metric(_):
        return 1 / 0

    result = ReadabilityAnalyzer.apply_metric(fake_metric, "this is a text")
    assert result == "N/A"


def test_textstat_metrics_flesh_reading_ease(content_example_1):
    res = textstat.flesch_reading_ease(content_example_1)
    assert math.isclose(res, 117.16)


def test_textstat_metrics_difficult_words(content_example_1):
    res = textstat.difficult_words(content_example_1)
    assert res == 0


def test_textstat_metrics_lexicon_count(content_example_1):
    res = textstat.lexicon_count(content_example_1)
    assert res == 5


def test_textstat_metrics_words_per_sentence(content_example_1):
    res = textstat.words_per_sentence(content_example_1)
    assert res == 5.0
