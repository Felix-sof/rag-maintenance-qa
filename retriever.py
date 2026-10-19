"""
Hybrid retrieval: dense vectors + BM25 keywords, fused with Reciprocal Rank
Fusion, optionally re-ranked by a cross-encoder.

Why not vectors alone? Embeddings are good at meaning ("makine ısınıp yavaş
dönüyor" -> heat dissipation failure) but weak at exact tokens - failure
codes like "OSF", acronyms like "LOTO", numbers. BM25 is the opposite.
Fusing their *ranks* (not their incomparable scores) gets the strengths of
both. A cross-encoder then reads the query and each candidate together,
which is more accurate than comparing two independently computed vectors,
but too slow to run over the whole corpus - so it only re-orders the fused
shortlist.

Modes (RETRIEVAL_MODE env var): vector | bm25 | hybrid | hybrid_rerank.
`python -m evals.retrieval_eval` compares all four. ("bm25" means keyword
search only, whichever keyword backend is in use.)

Keyword backends (KEYWORD_BACKEND env var):
    bm25            rank_bm25 in memory, over the stems below - works with either store
    pg_fts_turkish  Postgres full-text search, Snowball Turkish stemmer (VECTOR_BACKEND=pgvector)
    pg_fts_stems    Postgres full-text search over the same stems as bm25 (VECTOR_BACKEND=pgvector)
`python -m evals.retrieval_eval --compare-backends` measures them.

Every returned hit keeps `score` = cosine similarity to the query (computed
from the stored embedding even for keyword-only hits), so rag.py's cosine
gate means the same thing in every mode that uses it. In hybrid_rerank mode
the gate reads the cross-encoder's score (`ranks.rerank`) instead.
"""

import os
import re

from vector_store import MAX_TOP_K, KnowledgeBase, get_store

RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid_rerank")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
KEYWORD_BACKEND = os.environ.get("KEYWORD_BACKEND", "bm25")
MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
KEYWORD_BACKENDS = ("bm25", "pg_fts_turkish", "pg_fts_stems")

CANDIDATES = 20  # per retriever, before fusion
RERANK_POOL = 12  # fused candidates the cross-encoder re-scores
RRF_K = 60  # standard Reciprocal Rank Fusion constant (Cormack et al., 2009)
STEM_LENGTH = 5  # Turkish "F5" stemming: keep the first 5 characters of each word

STOPWORDS = {
    "ve", "ile", "bir", "bu", "şu", "o", "ne", "mi", "mı", "mu", "mü", "için", "da", "de",
    "ki", "gibi", "daha", "çok", "en", "olan", "nasıl", "neler", "nedir", "hangi", "var",
    "the", "a", "an", "of", "to", "in", "for", "is", "what", "how", "and", "or",
}


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless i rules ("I" -> "ı", "İ" -> "i")."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str) -> list[str]:
    """BM25 tokens: Turkish-lowercased words, stopwords removed, truncated to STEM_LENGTH.

    Turkish is agglutinative - "arıza", "arızası", "arızaları", "arızanın"
    are one word to a reader but four tokens to BM25. Truncating to the
    first five characters is a crude but well-studied stemmer for Turkish
    retrieval (Can et al., 2008) that needs no dictionary.
    """
    words = re.findall(r"\w+", turkish_lower(text))
    return [w[:STEM_LENGTH] for w in words if w not in STOPWORDS]


class CrossEncoderReranker:
    """Multilingual cross-encoder (mMiniLM fine-tuned on mMARCO), loaded lazily."""

    def __init__(self, model_name: str = RERANK_MODEL):
        self.name = model_name
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name)
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


class Retriever:
    def __init__(self, store: KnowledgeBase = None, mode: str = RETRIEVAL_MODE, reranker=None, keyword=KEYWORD_BACKEND):
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")
        if keyword not in KEYWORD_BACKENDS:
            raise ValueError(f"unknown keyword backend {keyword!r}; expected one of {KEYWORD_BACKENDS}")
        self.store = store or get_store()
        if keyword != "bm25" and not hasattr(self.store, "keyword_ranking"):
            raise ValueError(f"keyword backend {keyword!r} runs in Postgres: it needs VECTOR_BACKEND=pgvector")
        self.mode = mode
        self.keyword = keyword
        self.reranker = reranker
        self._corpus_fingerprint = None
        self._bm25 = None
        self._bm25_ids: list[str] = []

    def _refresh_bm25(self) -> None:
        """(Re)build the in-memory BM25 index whenever the vector index has been rebuilt."""
        fingerprint = self.store.index_fingerprint()
        if fingerprint == self._corpus_fingerprint:
            return
        from rank_bm25 import BM25Okapi

        chunks = self.store.all_chunks()
        self._bm25_ids = [c["id"] for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None
        self._corpus_fingerprint = fingerprint

    def _keyword_ranking(self, query: str) -> list[str]:
        if self.keyword != "bm25":
            return self.store.keyword_ranking(query, self.keyword, CANDIDATES)
        self._refresh_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._bm25_ids[i] for i in ranked[:CANDIDATES] if scores[i] > 0]

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        query_vector = self.store.embedder.embed_query(query)

        vector_ids = [h["id"] for h in self.store.search_by_vector(query_vector, CANDIDATES, max_k=CANDIDATES)]
        if not vector_ids:  # empty index
            return []
        keyword_ids = self._keyword_ranking(query) if self.mode != "vector" else []

        if self.mode == "vector":
            ranked = vector_ids
        elif self.mode == "bm25":
            ranked = keyword_ids
        else:
            fused: dict[str, float] = {}
            for ranking in (vector_ids, keyword_ids):
                for rank, chunk_id in enumerate(ranking, start=1):
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            ranked = sorted(fused, key=lambda cid: -fused[cid])

        # only the shortlist is fetched: text for the reranker, embedding for the cosine
        shortlist = ranked[:RERANK_POOL] if self.mode == "hybrid_rerank" else ranked[:top_k]
        chunks = self.store.get_chunks(shortlist)

        rerank_scores = {}
        if self.mode == "hybrid_rerank" and shortlist:
            reranker = self.reranker or _default_reranker()
            rerank_scores = dict(zip(shortlist, reranker.score(query, [chunks[cid]["text"] for cid in shortlist])))
            ranked = sorted(shortlist, key=lambda cid: -rerank_scores[cid])

        vector_rank = {cid: r for r, cid in enumerate(vector_ids, start=1)}
        keyword_rank = {cid: r for r, cid in enumerate(keyword_ids, start=1)}
        hits = []
        for chunk_id in ranked[:top_k]:
            chunk = chunks[chunk_id]
            cosine = sum(a * b for a, b in zip(query_vector, chunk["embedding"]))  # vectors are L2-normalized
            hits.append(
                {
                    "id": chunk_id,
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "doc_type": chunk["doc_type"],
                    "priority": chunk.get("priority", 1),
                    "updated": chunk.get("updated", ""),
                    "score": round(cosine, 4),
                    "text": chunk["text"],
                    "ranks": {
                        "vector": vector_rank.get(chunk_id),
                        "bm25": keyword_rank.get(chunk_id),
                        "rerank": round(rerank_scores[chunk_id], 4) if chunk_id in rerank_scores else None,
                    },
                }
            )
        return hits


_reranker: CrossEncoderReranker = None
_retriever: Retriever = None


def _default_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def get_retriever() -> Retriever:
    """Process-wide retriever over the process-wide store (tests replace it via retriever._retriever)."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
