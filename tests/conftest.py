"""Shared test fixtures.

Nothing in the test suite touches the network or downloads a model:
  - HashEmbedder replaces the sentence-transformers model with a
    deterministic bag-of-words hash (keyword overlap -> similarity),
  - FakeGenerator replaces Gemini and records the prompts it receives,
  - the process-wide store/generator are swapped for these on every test.
Retrieval quality with the real model is measured by evals/, not here.
"""

import hashlib
import math
import re

import pytest


class HashEmbedder:
    name = "test-hash-embedder"
    DIM = 256

    def _vec(self, text):
        v = [0.0] * self.DIM
        for token in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeGenerator:
    """Stands in for GeminiGenerator. `responses` are returned in order (last one repeats)."""

    def __init__(self, *responses, error=None):
        self.responses = list(responses) or ["Cevap [1]."]
        self.error = error
        self.calls = []  # [(system, prompt)]

    def generate(self, system, prompt, temperature=0.2):
        self.calls.append((system, prompt))
        if self.error:
            raise self.error
        return self.responses[min(len(self.calls), len(self.responses)) - 1]


class StubStore:
    """A retriever that returns fixed hits and records the queries it receives."""

    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, top_k=4):
        self.queries.append(query)
        return self.hits[:top_k]


def make_hit(score, source="failure_modes.md", section="Arızalar > HDF", doc_type="reference", text="HDF metni"):
    return {"source": source, "section": section, "doc_type": doc_type, "score": score, "text": text}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Point the app at a temp index of the real knowledge/ docs and a fake LLM."""
    import generator
    import retriever
    import vector_store

    store = vector_store.VectorStore(index_dir=tmp_path / "index", embedder=HashEmbedder())
    monkeypatch.setattr(vector_store, "_store", store)
    # hybrid without the cross-encoder: no model download in tests
    monkeypatch.setattr(retriever, "_retriever", retriever.Retriever(store=store, mode="hybrid"))
    monkeypatch.setattr(generator, "_generator", FakeGenerator())
    yield


def make_pdf(pages: list[str], title: str = None) -> bytes:
    """Build a minimal valid PDF with one Helvetica text line per input line (ASCII only).

    Avoids a PDF-writing dependency just for tests; pypdf extracts the text back.
    """
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in text.split("\n"):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    info = ""
    if title:
        objects.append(f"<< /Title ({title}) >>".encode("latin-1"))
        info = f" /Info {len(objects)} 0 R"

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R{info} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out
