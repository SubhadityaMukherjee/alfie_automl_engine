from pathlib import Path

import pytest

from app.automlplus.utils import ImageConverter

_HERE = Path(__file__).parent
_IMAGE_PATH = _HERE / "atq.png"


@pytest.fixture
def image_path():
    return str(_IMAGE_PATH)


@pytest.fixture
def image_bytes():
    return _IMAGE_PATH.read_bytes()


def test_to_base64(image_path: str):
    base64im = ImageConverter.to_base64(image_path)
    assert isinstance(base64im, str)
    assert "=" in base64im


def test_bytes_to_base64(image_bytes: bytes):
    base64im = ImageConverter.bytes_to_base64(image_bytes)
    assert isinstance(base64im, str)
    assert "=" in base64im
