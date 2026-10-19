"""
Persistent vector index over the knowledge base.

Two backends, chosen with VECTOR_BACKEND:
    chroma    (default) a local ChromaDB index in data/chroma - no server needed
    pgvector  PostgreSQL + pgvector at DATABASE_URL (pg_store.py)

Both store a fingerprint of everything the index depends on - the documents'
contents, the embedding model, and the chunk size. If any of them change,
the next build()/search() rebuilds the index automatically, so editing a
document never leaves stale vectors behind.

    python vector_store.py --rebuild        # (re)build the index
    python vector_store.py "HDF nedir?"     # retrieval only, no LLM
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from chunking import MAX_CHUNK_CHARS, META_SUFFIX, SUPPORTED_EXTENSIONS, document_paths, load_chunks, source_name
from embeddings import E5Embedder, Embedder

load_dotenv()

DOCS_DIR = Path(__file__).parent / "knowledge"
INDEX_DIR = Path(__file__).parent / "data" / "chroma"
COLLECTION_NAME = "knowledge_base"
MAX_TOP_K = 10
UPLOADS_SUBDIR = "uploads"  # documents added from the UI live here and can be deleted from it
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "chroma")
BACKENDS = ("chroma", "pgvector")


class KnowledgeBase:
    """What both backends share: the documents on disk, their fingerprint, uploads.

    A backend implements build(), skipped(), stats(), index_fingerprint(),
    all_chunks(), get_chunks() and search_by_vector().
    """

    def __init__(self, docs_dir: Path = DOCS_DIR, embedder: Embedder = None):
        self.docs_dir = Path(docs_dir)
        self.embedder = embedder or E5Embedder()

    def fingerprint(self) -> str:
        h = hashlib.sha256(f"{self.embedder.name}|{MAX_CHUNK_CHARS}".encode("utf-8"))
        for path in document_paths(self.docs_dir):
            h.update(source_name(path, self.docs_dir).encode("utf-8"))
            h.update(path.read_bytes())
            sidecar = path.with_name(path.name + META_SUFFIX)
            if sidecar.exists():
                h.update(sidecar.read_bytes())
        return h.hexdigest()

    def save_upload(self, filename: str, data: bytes, meta: dict) -> str:
        """Store an uploaded document (plus its metadata sidecar) under uploads/ and re-index.

        Returns the document's source name, e.g. "uploads/kilavuz.pdf".
        """
        safe = re.sub(r"[^\w.\-]+", "_", Path(filename).name).strip("._") or "belge"
        if Path(safe).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported file type: {filename} (supported: {', '.join(SUPPORTED_EXTENSIONS)})")
        target = self.docs_dir / UPLOADS_SUBDIR / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.with_name(target.name + META_SUFFIX).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.build()
        return source_name(target, self.docs_dir)

    def delete_upload(self, source: str) -> None:
        """Remove an uploaded document (only under uploads/) and re-index."""
        target = (self.docs_dir / source).resolve()
        uploads = (self.docs_dir / UPLOADS_SUBDIR).resolve()
        if uploads not in target.parents:
            raise ValueError(f"only uploaded documents can be deleted: {source}")
        target.unlink(missing_ok=True)
        target.with_name(target.name + META_SUFFIX).unlink(missing_ok=True)
        self.build()

    def chunk_counts(self) -> dict[str, int]:
        """Number of indexed chunks per document source."""
        counts: dict[str, int] = {}
        for chunk in self.all_chunks():
            counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
        return counts

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Return the top_k most similar chunks, best first, with cosine similarity as `score`."""
        return self.search_by_vector(self.embedder.embed_query(query), top_k)


class VectorStore(KnowledgeBase):
    """The ChromaDB backend."""

    def __init__(self, docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR, embedder: Embedder = None):
        import chromadb
        from chromadb.config import Settings

        super().__init__(docs_dir, embedder)
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))
        self._collection = None

    def build(self, force: bool = False) -> dict:
        """(Re)build the index if the documents/model changed since the last build, or if forced."""
        fingerprint = self.fingerprint()
        try:
            collection = self._client.get_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - the "not found" error type differs across chromadb versions
            collection = None

        if collection is not None:
            if not force and (collection.metadata or {}).get("fingerprint") == fingerprint:
                self._collection = collection
                return {"rebuilt": False, "chunks": collection.count(), "skipped": self.skipped()}
            self._client.delete_collection(COLLECTION_NAME)

        errors: dict[str, str] = {}
        chunks = load_chunks(self.docs_dir, errors=errors)
        collection = self._client.create_collection(
            COLLECTION_NAME,
            # chromadb metadata values must be scalars, so the skip report is stored as JSON
            metadata={"hnsw:space": "cosine", "fingerprint": fingerprint, "skipped": json.dumps(errors)},
        )
        if chunks:
            collection.add(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=self.embedder.embed_documents([c.text for c in chunks]),
                metadatas=[
                    {
                        "source": c.source,
                        "section": c.section,
                        "title": c.title,
                        "doc_type": c.doc_type,
                        "priority": c.priority,
                        "updated": c.updated,
                    }
                    for c in chunks
                ],
            )
        self._collection = collection
        return {"rebuilt": True, "chunks": len(chunks), "skipped": errors}

    def skipped(self) -> dict[str, str]:
        """Documents the last build couldn't read, with the reason."""
        return json.loads((self._get_collection().metadata or {}).get("skipped") or "{}")

    def _get_collection(self):
        if self._collection is None:
            self.build()
        return self._collection

    def stats(self) -> dict:
        collection = self._get_collection()
        metadatas = collection.get(include=["metadatas"])["metadatas"] if collection.count() else []
        return {
            "documents": len({m["source"] for m in metadatas}),
            "chunks": collection.count(),
            "embedding_model": self.embedder.name,
            "skipped": self.skipped(),
        }

    def index_fingerprint(self) -> str:
        """Fingerprint of the index currently loaded (changes whenever it is rebuilt)."""
        return (self._get_collection().metadata or {}).get("fingerprint", "")

    @staticmethod
    def _rows(res: dict) -> list[dict]:
        return [
            {"id": id_, "text": doc, "embedding": list(emb), **meta}
            for id_, doc, meta, emb in zip(res["ids"], res["documents"], res["metadatas"], res["embeddings"])
        ]

    def all_chunks(self) -> list[dict]:
        """Every indexed chunk with its stored embedding (used to build the keyword index)."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        return self._rows(collection.get(include=["documents", "metadatas", "embeddings"]))

    def get_chunks(self, ids: list[str]) -> dict[str, dict]:
        """The chunks with these ids, with their stored embeddings, keyed by id."""
        if not ids:
            return {}
        res = self._get_collection().get(ids=list(ids), include=["documents", "metadatas", "embeddings"])
        return {row["id"]: row for row in self._rows(res)}

    def search_by_vector(self, query_vector: list[float], top_k: int = 4, max_k: int = MAX_TOP_K) -> list[dict]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        top_k = max(1, min(int(top_k), max_k, collection.count()))
        res = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": id_,
                "source": meta["source"],
                "section": meta["section"],
                "doc_type": meta["doc_type"],
                "priority": meta.get("priority", 1),
                "updated": meta.get("updated", ""),
                "score": round(1 - dist, 4),
                "text": doc,
            }
            for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


def make_store(backend: str = VECTOR_BACKEND, **kwargs) -> KnowledgeBase:
    if backend == "chroma":
        return VectorStore(**kwargs)
    if backend == "pgvector":
        from pg_store import PgVectorStore

        return PgVectorStore(**kwargs)
    raise ValueError(f"unknown VECTOR_BACKEND {backend!r}; expected one of {BACKENDS}")


_store: KnowledgeBase = None


def get_store() -> KnowledgeBase:
    """Process-wide store, created lazily (tests replace it via vector_store._store)."""
    global _store
    if _store is None:
        _store = make_store()
    return _store


if __name__ == "__main__":
    store = get_store()
    if "--rebuild" in sys.argv:
        print(store.build(force=True))
        print(store.stats())
        sys.exit(0)

    query = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("--"))
    if not query:
        print(__doc__)
        sys.exit(0)
    for hit in store.search(query):
        print(f"{hit['score']:.3f}  {hit['source']} > {hit['section']}")
