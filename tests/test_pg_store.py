"""Tests for the pgvector backend against a real PostgreSQL + pgvector.

Skipped unless TEST_DATABASE_URL is set (CI runs a pgvector service
container; locally `python -m db.local_postgres` prints a URL). Each test
gets its own Postgres schema, dropped afterwards.
"""

import os
import uuid

import pytest

from conftest import HashEmbedder
from retriever import Retriever

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")

FAILURES = (
    "---\ntitle: Arızalar\ntype: reference\npriority: 3\nupdated: 2026-09-01\nsource: AI4I belgesi\n---\n\n"
    "# Arızalar\n\n## HDF\n\nISI dağıtım arızası düşük devirde oluşur.\n\n"
    "## OSF\n\nAşırı zorlanma arızası OSF kodlu arızadır.\n\n"
    "## PWF\n\nGüç arızası tork ve devir çarpımı sınır dışına çıkınca oluşur.\n"
)


@pytest.fixture
def make_store(tmp_path):
    import psycopg
    from psycopg import sql

    from pg_store import PgVectorStore

    schemas, stores = [], []

    def make(docs=None, embedder=None):
        docs_dir = tmp_path / f"docs{len(stores)}"
        docs_dir.mkdir()
        for name, text in (docs or {"arizalar.md": FAILURES}).items():
            (docs_dir / name).write_text(text, encoding="utf-8")
        schemas.append(f"test_{uuid.uuid4().hex[:12]}")
        store = PgVectorStore(docs_dir, DATABASE_URL, embedder or HashEmbedder(dim=384), schema=schemas[-1])
        stores.append(store)
        return store

    yield make
    for store in stores:
        store.close()
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        for schema in schemas:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def _sql(store, query):
    return store._query(query)


def test_build_indexes_every_section_and_skips_an_unchanged_rebuild(make_store):
    store = make_store()
    assert store.build() == {"rebuilt": True, "chunks": 3, "skipped": {}}
    assert store.stats()["documents"] == 1
    assert store.build()["rebuilt"] is False

    (store.docs_dir / "rnf.md").write_text("# RNF\n\nRNF rastgele arızadır.\n", encoding="utf-8")
    assert store.build() == {"rebuilt": True, "chunks": 4, "skipped": {}}
    assert store.chunk_counts() == {"arizalar.md": 3, "rnf.md": 1}


def test_documents_keep_their_metadata_and_raw_front_matter(make_store):
    store = make_store()
    store.build()
    row = _sql(store, "SELECT title, doc_type, priority, updated, metadata->>'source' FROM documents")[0]
    assert row == ("Arızalar", "reference", 3, "2026-09-01", "AI4I belgesi")
    assert _sql(store, "SELECT position, section FROM chunks ORDER BY position")[0] == (0, "Arızalar > HDF")


def test_vector_search_returns_cosine_scores_best_first(make_store):
    store = make_store()
    hits = store.search("OSF kodlu arıza", top_k=3)
    assert hits[0]["section"] == "Arızalar > OSF"
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert hits[0]["priority"] == 3


def test_get_chunks_returns_text_and_embedding_by_id(make_store):
    store = make_store()
    first = store.search("OSF", top_k=1)[0]
    chunk = store.get_chunks([first["id"]])[first["id"]]
    assert chunk["text"] == first["text"]
    assert len(chunk["embedding"]) == 384


class TestKeywordSearch:
    def test_stems_match_inflected_words(self, make_store):
        store = make_store()
        # "arızaları" and "arızası" share the stem "arıza"; "oluşu" is in HDF and PWF only
        ids = store.keyword_ranking("arızaların oluşumu", "pg_fts_stems", 10)
        sections = {c["section"] for c in store.get_chunks(ids).values()}
        assert sections == {"Arızalar > HDF", "Arızalar > OSF", "Arızalar > PWF"}

    def test_bare_code_ranks_its_section_first(self, make_store):
        store = make_store()
        for backend in ("pg_fts_stems", "pg_fts_turkish"):
            ids = store.keyword_ranking("OSF", backend, 10)
            assert store.get_chunks(ids[:1])[ids[0]]["section"] == "Arızalar > OSF", backend

    def test_turkish_capitals_are_lowercased_before_postgres(self, make_store):
        # the document says "ISI": Postgres alone would index it as "isi", not "ısı"
        store = make_store()
        for backend in ("pg_fts_stems", "pg_fts_turkish"):
            ids = store.keyword_ranking("ısı", backend, 10)
            assert [store.get_chunks(ids)[i]["section"] for i in ids] == ["Arızalar > HDF"], backend

    def test_stopword_only_query_matches_nothing(self, make_store):
        store = make_store()  # PWF's text has "ve"; both configurations drop it as a stopword
        for backend in ("pg_fts_stems", "pg_fts_turkish"):
            assert store.keyword_ranking("ve ile", backend, 10) == [], backend


@pytest.mark.parametrize("keyword", ["bm25", "pg_fts_turkish", "pg_fts_stems"])
def test_hybrid_retrieval_over_postgres(make_store, keyword):
    store = make_store()
    hits = Retriever(store=store, mode="hybrid", keyword=keyword).search("OSF arızası", top_k=3)
    assert hits[0]["section"] == "Arızalar > OSF"
    assert hits[0]["ranks"]["vector"] and hits[0]["ranks"]["bm25"]


def test_uploads_are_indexed_and_deleted(make_store):
    store = make_store()
    source = store.save_upload("pompa notu.md", "# Pompa\n\nPompa basıncını kontrol edin.\n".encode(), {"type": "example"})
    assert store.chunk_counts()[source] == 1
    store.delete_upload(source)
    assert source not in store.chunk_counts()


def test_migrations_run_once(make_store):
    from db.migrate import migrate

    store = make_store()
    store.build()
    assert migrate(store._connection()) == []
    assert _sql(store, "SELECT version FROM schema_migrations") == [("001_pgvector_index",)]


def test_an_embedding_of_the_wrong_size_is_refused(make_store):
    store = make_store(embedder=HashEmbedder(dim=256))
    with pytest.raises(ValueError, match="vector\\(384\\)"):
        store.build()
