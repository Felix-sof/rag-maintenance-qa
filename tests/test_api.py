import pytest
from fastapi.testclient import TestClient

import api
import rag


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health_reports_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index"]["documents"] == 5


def test_search_returns_ranked_hits(client):
    body = client.get("/search", params={"q": "takım değişimi", "top_k": 3}).json()
    assert len(body["results"]) == 3
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_ask_passes_history_and_top_k(client, monkeypatch):
    seen = {}

    def fake_answer(question, history, top_k):
        seen.update(question=question, history=history, top_k=top_k)
        return {"answer": "ok"}

    monkeypatch.setattr(rag, "answer", fake_answer)
    resp = client.post(
        "/ask",
        json={"question": "Peki adımları?", "history": [{"question": "q", "answer": "a"}], "top_k": 2},
    )
    assert resp.status_code == 200
    assert seen == {"question": "Peki adımları?", "history": [{"question": "q", "answer": "a"}], "top_k": 2}


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x", "top_k": 99}, {}])
def test_ask_rejects_invalid_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422
