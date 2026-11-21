from fastapi.testclient import TestClient

from src.api import app

# No `with` block, so the startup lifespan (index loading) is skipped.
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_empty_question_rejected():
    assert client.post("/ask", json={"query": "   "}).status_code == 400


def test_too_long_question_rejected():
    assert client.post("/ask", json={"query": "a" * 501}).status_code == 400


def test_missing_field_rejected():
    assert client.post("/ask", json={}).status_code == 422
    