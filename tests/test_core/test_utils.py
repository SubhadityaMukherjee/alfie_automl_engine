"""Tests for app.core.utils.render_template."""

from unittest.mock import MagicMock

import pytest
from jinja2 import Environment, FileSystemLoader

from app.core.utils import render_template


@pytest.fixture
def jinja_env():
    """Real Jinja2 environment pointing at the project's prompt templates."""
    return Environment(loader=FileSystemLoader("app/core/prompt_templates"))


def test_render_template_success(jinja_env):
    result = render_template(
        jinja_env,
        "build_chunk_prompt.txt",
        filename="test.html",
        chunk="<p>Hello</p>",
        idx=0,
        total=1,
        start_line=1,
        end_line=10,
    )
    assert isinstance(result, str)
    assert "test.html" in result
    assert "<p>Hello</p>" in result


def test_render_template_missing_template_raises(jinja_env):
    with pytest.raises(RuntimeError, match="Failed to load template"):
        render_template(jinja_env, "nonexistent_template_xyz.txt")


def test_render_template_render_error_raises():
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_template.render.side_effect = Exception("render failed")
    mock_env.get_template.return_value = mock_template

    with pytest.raises(RuntimeError, match="Failed to render template"):
        render_template(mock_env, "some_template.txt", foo="bar")


def test_render_template_passes_kwargs():
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_template.render.return_value = "rendered"
    mock_env.get_template.return_value = mock_template

    result = render_template(mock_env, "template.txt", key1="val1", key2="val2")
    mock_template.render.assert_called_once_with(key1="val1", key2="val2")
    assert result == "rendered"
