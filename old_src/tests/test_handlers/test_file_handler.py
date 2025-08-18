import io
import os
import tempfile
from pathlib import Path

import pytest
from file_handler import FileHandler


@pytest.fixture
def txt_file():
    content = b"Hello\nThis is a test file."
    file = io.BytesIO(content)
    file.name = "test.txt"
    return file


@pytest.fixture
def csv_file():
    content = b"name,age\nAlice,30\nBob,25"
    file = io.BytesIO(content)
    file.name = "test.csv"
    return file


@pytest.fixture
def docx_file():
    from docx import Document

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        temp_path = f.name
    doc = Document()
    doc.add_paragraph("This is a DOCX test.")
    doc.save(temp_path)

    file = open(temp_path, "rb")
    yield file
    file.close()
    Path(temp_path).unlink(missing_ok=True)


def test_read_txt_file(txt_file):
    filename, mime, path = FileHandler.save_temp_file(txt_file)
    content = FileHandler.read_file_content(path, mime)
    assert "Hello" in content
    path.unlink(missing_ok=True)


def test_read_csv_file(csv_file):
    filename, mime, path = FileHandler.save_temp_file(csv_file)
    content = FileHandler.read_file_content(path, mime)
    assert content == "name,age\nAlice,30\nBob,25"
    path.unlink(missing_ok=True)


def test_read_docx_file(docx_file):
    filename, mime, path = FileHandler.save_temp_file(docx_file)
    content = FileHandler.read_file_content(path, mime)
    assert "DOCX test" in content
    path.unlink(missing_ok=True)


def test_unsupported_file_type():
    file = io.BytesIO(b"%PDF-1.4 fake content")
    file.name = "test.pdf"
    filename, mime, path = FileHandler.save_temp_file(file)
    content = FileHandler.read_file_content(path, mime)
    assert "unsupported file type" in content
    path.unlink(missing_ok=True)


def test_aggregate_file_content_and_paths(txt_file):
    content, paths = FileHandler.aggregate_file_content_and_paths([txt_file])
    for p in paths.values():
        Path(p).unlink(missing_ok=True)
    assert "test.txt" in content
