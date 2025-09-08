import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.website_accessibility.main import app as website_app

client = TestClient(website_app)
TEST_HTML_FILE = Path(__file__).parent.parent / "test_data" / "test.html"
IMAGE_TEST = "https://media.cnn.com/api/v1/images/stellar/prod/160107100400-monkey-selfie.jpg?q=w_1160,c_fill/f_webp"


class TestReadabilityEndpoint:
    @pytest.fixture
    def resp_200(self):
        test_html_content = open(TEST_HTML_FILE, "rb").read()
        return client.post(
            "/web_access/readability/",
            files={"file": ("test.html", test_html_content, "text/html")},
        )

    @pytest.fixture
    def resp_500(self):
        return client.post(
            "/web_access/readability/",
            files={"file": ("test.html", b"", "text/html")},
        )

    def test_status_code_200(self, resp_200):
        assert resp_200.status_code == 200

    def test_status_code_500(self, resp_500):
        assert resp_500.status_code == 500

    def test_payload_fields(self, resp_200):
        data = resp_200.json()
        assert data["Flesch Reading Ease"] > 0
        assert data["Difficult Words"] > 0
        assert data["Lexicon Count"] > 0
        assert data["Avg Sentence Length"] > 0


@pytest.mark.full
class TestAccessibilityEndpoint:
    @pytest.fixture
    def resp(self):
        test_html_content = open(TEST_HTML_FILE, "rb").read()
        return client.post(
            "/web_access/accessibility/",
            files={"file": ("test.html", test_html_content, "text/html")},
        )

    def test_status_code_200(self, resp):
        assert resp.status_code == 200

    def test_streamed_payload_shape(self, resp):
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().splitlines() if line.strip()]
        assert len(lines) >= 1
        summary_obj = json.loads(lines[-1])
        assert "average_score" in summary_obj
        for line in lines[:-1]:
            obj = json.loads(line)
            assert "chunk" in obj
            assert "start_line" in obj
            assert "end_line" in obj


@pytest.mark.full
class TestAltTextEndpoint:
    @pytest.fixture
    def resp_ok(self):
        return client.post(
            "/web_access/check-alt-text/",
            data={
                "image_url": IMAGE_TEST,
                "alt_text": "close-up of a monkey with a wide-open mouth",
            },
        )

    @pytest.fixture
    def resp_bad(self):
        return client.post(
            "/web_access/check-alt-text/",
            data={
                "image_url": IMAGE_TEST,
                "alt_text": "close-up of a nintendo switch",
            },
        )

    def test_status_code_200(self, resp_ok):
        assert resp_ok.status_code == 200

    def test_correct_alt_text_payload(self, resp_ok):
        data = resp_ok.json()
        assert "monkey" in data["evaluation"]
        assert data["src"] != ""
        assert data["alt_text"] != ""

    def test_incorrect_alt_text_payload(self, resp_bad):
        data = resp_bad.json()
        assert "No" in data["evaluation"]
        assert data["src"] != ""
        assert data["alt_text"] != ""
