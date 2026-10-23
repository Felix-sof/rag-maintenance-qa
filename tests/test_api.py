import pytest
from fastapi.testclient import TestClient

import api
import rag


@pytest.fixture
def client(monkeypatch):
    import rate_limit

    # fresh counters per test: the limits are per process, and the tests share one
    monkeypatch.setattr(api, "limiter", rate_limit.RateLimiter())
    monkeypatch.setattr(api, "light_limiter", rate_limit.RateLimiter(per_minute=60, per_day=5_000))
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


def test_search_can_compare_modes(client):
    body = client.get("/search", params={"q": "OSF", "mode": "bm25", "top_k": 1}).json()
    assert body["mode"] == "bm25"
    assert body["results"][0]["ranks"]["bm25"] == 1
    assert "osf" in body["results"][0]["text"].lower()
    assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422


def test_ask_passes_history_and_top_k(client, monkeypatch):
    seen = {}

    def fake_answer(question, history, top_k, tools=None):
        seen.update(question=question, history=history, top_k=top_k, tools=tools)
        return {"answer": "ok"}

    monkeypatch.setattr(rag, "answer", fake_answer)
    resp = client.post(
        "/ask",
        json={"question": "Peki adımları?", "history": [{"question": "q", "answer": "a"}], "top_k": 2},
    )
    assert resp.status_code == 200
    assert seen == {
        "question": "Peki adımları?", "history": [{"question": "q", "answer": "a"}], "top_k": 2, "tools": api.copilot_tools,
    }


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x", "top_k": 99}, {}])
def test_ask_rejects_invalid_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_upload_list_and_delete_a_document(client, monkeypatch, tmp_path):
    import shutil

    import vector_store
    from conftest import HashEmbedder, make_pdf

    docs = tmp_path / "docs"
    shutil.copytree(vector_store.DOCS_DIR, docs)
    store = vector_store.VectorStore(docs_dir=docs, index_dir=tmp_path / "idx2", embedder=HashEmbedder())
    monkeypatch.setattr(vector_store, "_store", store)

    resp = client.post(
        "/documents",
        files={"file": ("pompa.pdf", make_pdf(["Pump pressure check"]), "application/pdf")},
        data={"doc_type": "reference", "priority": "3", "updated": "2026-09-01"},
    )
    assert resp.status_code == 201
    assert resp.json() == {"source": "uploads/pompa.pdf", "chunks": 1, "error": None}
    assert client.get("/documents").json()["documents"]["uploads/pompa.pdf"] == 1

    assert client.delete("/documents/uploads/pompa.pdf").status_code == 200
    assert "uploads/pompa.pdf" not in client.get("/documents").json()["documents"]


def test_upload_rejects_unsupported_type(client):
    resp = client.post("/documents", files={"file": ("rapor.docx", b"x", "application/octet-stream")})
    assert resp.status_code == 400


def test_built_in_documents_cannot_be_deleted(client):
    assert client.delete("/documents/safety.md").status_code == 400


def test_ask_is_rate_limited(client, monkeypatch):
    import rate_limit

    monkeypatch.setattr(api, "limiter", rate_limit.RateLimiter(per_minute=2, per_day=100))
    monkeypatch.setattr(rag, "answer", lambda question, history, top_k, tools=None: {"answer": "ok"})
    codes = [client.post("/ask", json={"question": "HDF?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert int(client.post("/ask", json={"question": "HDF?"}).headers["Retry-After"]) > 0


def test_read_only_deployment_refuses_uploads_and_deletes(client, monkeypatch):
    monkeypatch.setenv("READ_ONLY", "1")
    upload = client.post("/documents", files={"file": ("notlar.md", b"# Not", "text/markdown")})
    assert upload.status_code == 403
    assert client.delete("/documents/uploads/notlar.md").status_code == 403
    assert client.get("/health").json()["read_only"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "x" * 501},
        {"question": "HDF?", "history": [{"question": "q", "answer": "a"}] * 5},
        {"question": "HDF?", "history": [{"question": "q", "answer": "a" * 2001}]},
    ],
)
def test_oversized_input_is_rejected_before_any_work(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_search_has_a_looser_limit_and_a_size_cap(client, monkeypatch):
    import rate_limit

    monkeypatch.setattr(api, "light_limiter", rate_limit.RateLimiter(per_minute=2, per_day=100))
    assert client.get("/search", params={"q": "OSF"}).status_code == 200
    assert client.get("/search", params={"q": "x" * 501}).status_code == 422
    # the limit is checked before the input, so rejected requests count too: garbage isn't free
    assert client.get("/search", params={"q": "OSF"}).status_code == 429


def test_daily_cap_is_also_read_from_the_query_log(client, monkeypatch):
    """A restart clears the in-memory counter; the logged answers of the last 24 h still count."""
    import query_log

    class LoggedToday:
        def llm_answers_last_24h(self, source):
            return 20 if source == "api" else 0

    monkeypatch.setattr(api, "PER_DAY", 20)
    monkeypatch.setattr(query_log, "enabled", lambda: True)
    monkeypatch.setattr(query_log, "get", lambda: LoggedToday())
    monkeypatch.setattr(rag, "answer", lambda question, history, top_k, tools=None: {"answer": "ok"})
    resp = client.post("/ask", json={"question": "HDF?"})
    assert resp.status_code == 429 and resp.headers["Retry-After"] == "3600"


def test_health_does_not_query_the_index_on_every_call(client, monkeypatch):
    calls = []
    monkeypatch.setattr(api, "_health_cache", {"at": float("-inf"), "index": None})
    monkeypatch.setattr(api, "get_store", lambda: type("S", (), {"stats": lambda self: calls.append(1) or {}})())
    for _ in range(5):
        assert client.get("/health").status_code == 200
    assert calls == [1]


def test_the_copilot_tools_can_be_switched_off(client, monkeypatch):
    seen = {}
    monkeypatch.setenv("COPILOT", "0")
    monkeypatch.setattr(rag, "answer", lambda question, history, top_k, tools=None: seen.update(tools=tools) or {})
    client.post("/ask", json={"question": "HDF?"})
    assert seen == {"tools": None}
