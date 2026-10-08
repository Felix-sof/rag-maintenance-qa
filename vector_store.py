"""
Persistent vector index over the knowledge base (ChromaDB, cosine distance).

The index stores a fingerprint of everything it depends on - the documents'
contents, the embedding model, and the chunk size. If any of them change,
the next build()/search() rebuilds the index automatically, so editing a
document never leaves stale vectors behind.

    python vector_store.py --rebuild        # (re)build the index
    python vector_store.py "HDF nedir?"     # retrieval only, no LLM
"""

import hashlib
import sys
from pathlib import Path

from chunking import MAX_CHUNK_CHARS, document_paths, load_chunks
from embeddings import E5Embedder, Embedder

DOCS_DIR = Path(__file__).parent / "knowledge"
INDEX_DIR = Path(__file__).parent / "data" / "chroma"
COLLECTION_NAME = "knowledge_base"
MAX_TOP_K = 10


class VectorStore:
    def __init__(self, docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR, embedder: Embedder = None):
        import chromadb
        from chromadb.config import Settings

        self.docs_dir = Path(docs_dir)
        self.embedder = embedder or E5Embedder()
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))
        self._collection = None

    def fingerprint(self) -> str:
        h = hashlib.sha256(f"{self.embedder.name}|{MAX_CHUNK_CHARS}".encode("utf-8"))
        for path in document_paths(self.docs_dir):
            h.update(path.name.encode("utf-8"))
            h.update(path.read_bytes())
        return h.hexdigest()

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
                return {"rebuilt": False, "chunks": collection.count()}
            self._client.delete_collection(COLLECTION_NAME)

        chunks = load_chunks(self.docs_dir)
        collection = self._client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine", "fingerprint": fingerprint}
        )
        if chunks:
            collection.add(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=self.embedder.embed_documents([c.text for c in chunks]),
                metadatas=[
                    {"source": c.source, "section": c.section, "title": c.title, "doc_type": c.doc_type}
                    for c in chunks
                ],
            )
        self._collection = collection
        return {"rebuilt": True, "chunks": len(chunks)}

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
        }

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Return the top_k most similar chunks, best first, with cosine similarity as `score`."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        top_k = max(1, min(int(top_k), MAX_TOP_K, collection.count()))
        res = collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "source": meta["source"],
                "section": meta["section"],
                "doc_type": meta["doc_type"],
                "score": round(1 - dist, 4),
                "text": doc,
            }
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


_store: VectorStore = None


def get_store() -> VectorStore:
    """Process-wide store, created lazily (tests replace it via vector_store._store)."""
    global _store
    if _store is None:
        _store = VectorStore()
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
