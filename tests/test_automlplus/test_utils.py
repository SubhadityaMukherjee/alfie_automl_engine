import pytest
import requests

from app.automlplus.utils import ImageConverter


@pytest.fixture
def image_url():
    return "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Ataquechocrane.png/500px-Ataquechocrane.png"


@pytest.fixture
def image_bytes():
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Ataquechocrane.png/500px-Ataquechocrane.png"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageConverter/1.0)"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    if "image" not in resp.headers.get("Content-Type", ""):
        raise ValueError(f"URL does not point to an image: {url}")
    return resp.content


def test_to_base64(image_url: str):
    base64im = ImageConverter.to_base64(image_url)
    assert isinstance(base64im, str)
    assert "=" in base64im


def test_bytes_to_image64(image_bytes: str):
    assert isinstance(image_bytes, bytes)
