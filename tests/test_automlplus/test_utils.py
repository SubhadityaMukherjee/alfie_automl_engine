from pathlib import Path

import pytest

from app.automlplus.utils import ImageConverter

_HERE = Path(__file__).parent
_IMAGE_PATH = _HERE / "atq.png"


@pytest.fixture
def image_url():
    return "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Ataquechocrane.png/500px-Ataquechocrane.png"


@pytest.fixture
def image_bytes():
    return _IMAGE_PATH.read_bytes()


def test_to_base64(image_url: str):
    base64im = ImageConverter.to_base64(image_url)
    assert isinstance(base64im, str)
    assert "=" in base64im


def test_bytes_to_image64(image_bytes: bytes):
    assert isinstance(image_bytes, bytes)
