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


def test_search_can_compare_modes(client):
    body = client.get("/search", params={"q": "OSF", "mode": "bm25", "top_k": 1}).json()
    assert body["mode"] == "bm25"
    assert body["results"][0]["ranks"]["bm25"] == 1
    assert "osf" in body["results"][0]["text"].lower()
    assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422


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
