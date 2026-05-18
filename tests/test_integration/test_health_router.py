"""Integration tests for the shared health router."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.health import router as health_router

app = FastAPI()
app.include_router(health_router)

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_ready_endpoint():
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}
