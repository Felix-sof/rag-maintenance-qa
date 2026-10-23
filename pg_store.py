"""
PostgreSQL + pgvector backend: VECTOR_BACKEND=pgvector, DATABASE_URL=postgresql://...

The same contract as the ChromaDB backend in vector_store.py (build, search,
fingerprint, uploads), with the index in three tables (db/migrations/), plus
one thing Chroma can't do: keyword_ranking(), a full-text search inside the
database, so a hybrid query doesn't need the corpus in the app's memory.

Turkish text is lowercased in Python before it reaches Postgres. Postgres
lowercases with the database's locale: under the C locale `lower('ISI İŞLEM')`
is 'isi İŞlem' - the dotless ı is lost and non-ASCII capitals aren't touched.
turkish_lower() gives the same result whatever locale the server runs with.

Schema changes go through db/migrate.py, which runs on the first connection.
"""

import os
import re

import numpy as np
from dotenv import load_dotenv

import db
from chunking import document_metadata, load_chunks
from embeddings import Embedder
from retriever import tokenize, turkish_lower
from vector_store import DOCS_DIR, MAX_TOP_K, KnowledgeBase

load_dotenv()

BUILD_LOCK_ID = 7_301_002  # advisory lock: one rebuild at a time, across processes
TS_RANK_NORMALIZATION = 0  # ts_rank's document-length normalization flag (0 = none)

# keyword backend -> (text search configuration, tsvector column)
KEYWORD_SEARCH = {
    "pg_fts_turkish": ("turkish", "tsv_turkish"),  # Postgres's Snowball stemmer and stopwords
    "pg_fts_stems": ("simple", "tsv_stems"),  # retriever.tokenize(): 5-character stems, our stopwords
}

CHUNK_COLUMNS = """
    c.id, c.text, c.embedding, c.source, c.section, d.title, d.doc_type, d.priority, d.updated
"""


class PgVectorStore(KnowledgeBase):
    def __init__(self, docs_dir=DOCS_DIR, database_url: str = None, embedder: Embedder = None, schema: str = None):
        """`schema` keeps the tables in their own Postgres schema (tests use one per test)."""
        super().__init__(docs_dir, embedder)
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("VECTOR_BACKEND=pgvector needs DATABASE_URL (postgresql://user:password@host/db)")
        self.schema = schema
        self._conn = None
        self._built = False

    # --- connection -------------------------------------------------------

    def _connect(self):
        from pgvector.psycopg import register_vector

        conn = db.connect(self.database_url, schema=self.schema)
        register_vector(conn)  # numpy arrays <-> the vector type
        return conn

    def _connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
        return self._conn

    def _query(self, query, params=None) -> list[tuple]:
        """Run a read query, reconnecting once if the server dropped the connection.

        A serverless Postgres (Neon) suspends an idle database and closes its
        connections; a read is safe to repeat on a fresh one.
        """
        import psycopg

        try:
            return self._connection().execute(query, params).fetchall()
        except psycopg.OperationalError:
            self._conn = None
            return self._connection().execute(query, params).fetchall()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- building the index ----------------------------------------------

    def _stored_fingerprint(self) -> str:
        rows = self._query("SELECT fingerprint FROM index_state WHERE id = 1")
        return rows[0][0] if rows else ""

    def _ready(self) -> None:
        if not self._built:
            self.build()

    def build(self, force: bool = False) -> dict:
        """(Re)build the index if the documents/model changed since the last build, or if forced.

        The new rows replace the old ones in one transaction, so a reader
        sees either the old index or the new one, never a half-written mix.
        """
        fingerprint = self.fingerprint()
        if not force and self._stored_fingerprint() == fingerprint:
            self._built = True
            return {"rebuilt": False, "chunks": self._count(), "skipped": self.skipped()}

        # chunk and embed before taking the lock: the slow part shouldn't hold a transaction open
        errors: dict[str, str] = {}
        chunks = load_chunks(self.docs_dir, errors=errors)
        embeddings = self.embedder.embed_documents([c.text for c in chunks]) if chunks else []

        conn = self._connection()
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (BUILD_LOCK_ID,))
            row = conn.execute("SELECT fingerprint FROM index_state WHERE id = 1").fetchone()
            if not force and row and row[0] == fingerprint:  # another process built it while we embedded
                self._built = True
                return {"rebuilt": False, "chunks": self._count(), "skipped": self.skipped()}
            if embeddings:
                self._check_dimension(conn, len(embeddings[0]))

            conn.execute("DELETE FROM documents")  # chunks go with them (ON DELETE CASCADE)
            self._insert(conn, chunks, embeddings)
            conn.execute(
                "INSERT INTO index_state (id, fingerprint, skipped) VALUES (1, %s, %s)"
                " ON CONFLICT (id) DO UPDATE SET fingerprint = EXCLUDED.fingerprint,"
                " skipped = EXCLUDED.skipped, built_at = now()",
                (fingerprint, _jsonb(errors)),
            )
        self._built = True
        return {"rebuilt": True, "chunks": len(chunks), "skipped": errors}

    @staticmethod
    def _check_dimension(conn, size: int) -> None:
        column = conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        ).fetchone()[0]
        if column != size:
            raise ValueError(
                f"the embedding model makes {size}-dimensional vectors but chunks.embedding is vector({column}); "
                "add a migration that changes the column for this model"
            )

    def _insert(self, conn, chunks, embeddings) -> None:
        documents, positions = {}, {}
        for chunk in chunks:
            if chunk.source not in documents:
                raw = document_metadata(self.docs_dir / chunk.source)
                documents[chunk.source] = (
                    chunk.source, chunk.title, chunk.doc_type, chunk.priority, chunk.updated, _jsonb(raw)
                )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents (source, title, doc_type, priority, updated, metadata)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                list(documents.values()),
            )
            rows = []
            for chunk, embedding in zip(chunks, embeddings):
                positions[chunk.source] = positions.get(chunk.source, -1) + 1
                rows.append(
                    (
                        chunk.id, chunk.source, positions[chunk.source], chunk.section, chunk.text,
                        np.asarray(embedding, dtype=np.float32),
                        turkish_lower(chunk.text), " ".join(tokenize(chunk.text)),
                    )
                )
            cur.executemany(
                "INSERT INTO chunks (id, source, position, section, text, embedding, tsv_turkish, tsv_stems)"
                " VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('turkish', %s), to_tsvector('simple', %s))",
                rows,
            )

    # --- reading the index -------------------------------------------------

    def _count(self) -> int:
        return self._query("SELECT count(*) FROM chunks")[0][0]

    def skipped(self) -> dict[str, str]:
        """Documents the last build couldn't read, with the reason."""
        self._ready()
        rows = self._query("SELECT skipped FROM index_state WHERE id = 1")
        return rows[0][0] if rows else {}

    def stats(self) -> dict:
        self._ready()
        documents, chunks = self._query("SELECT count(DISTINCT source), count(*) FROM chunks")[0]
        return {
            "documents": documents,
            "chunks": chunks,
            "embedding_model": self.embedder.name,
            "skipped": self.skipped(),
            "backend": "pgvector",
        }

    def index_fingerprint(self) -> str:
        """Fingerprint of the index in the database (changes whenever it is rebuilt)."""
        self._ready()
        return self._stored_fingerprint()

    def chunk_counts(self) -> dict[str, int]:
        self._ready()
        return dict(self._query("SELECT source, count(*) FROM chunks GROUP BY source"))

    @staticmethod
    def _chunk(row: tuple) -> dict:
        id_, text, embedding, source, section, title, doc_type, priority, updated = row
        return {
            "id": id_, "text": text, "embedding": embedding.to_list(), "source": source, "section": section,
            "title": title, "doc_type": doc_type, "priority": priority, "updated": updated,
        }

    def all_chunks(self) -> list[dict]:
        """Every indexed chunk with its stored embedding (used to build the in-memory BM25 index)."""
        self._ready()
        rows = self._query(
            f"SELECT {CHUNK_COLUMNS} FROM chunks c JOIN documents d USING (source) ORDER BY c.source, c.position"
        )
        return [self._chunk(row) for row in rows]

    def get_chunks(self, ids: list[str]) -> dict[str, dict]:
        """The chunks with these ids, with their stored embeddings, keyed by id."""
        if not ids:
            return {}
        self._ready()
        rows = self._query(
            f"SELECT {CHUNK_COLUMNS} FROM chunks c JOIN documents d USING (source) WHERE c.id = ANY(%s)",
            (list(ids),),
        )
        return {row[0]: self._chunk(row) for row in rows}

    def search_by_vector(self, query_vector: list[float], top_k: int = 4, max_k: int = MAX_TOP_K) -> list[dict]:
        """Nearest chunks by cosine distance, best first, with cosine similarity as `score`.

        The ORDER BY ... LIMIT runs on chunks alone, the shape the HNSW index
        can serve; the join for the document's metadata comes after.
        """
        self._ready()
        top_k = max(1, min(int(top_k), max_k))
        rows = self._query(
            """
            WITH nearest AS (
                SELECT id, embedding <=> %(q)s AS distance
                FROM chunks
                ORDER BY distance
                LIMIT %(k)s
            )
            SELECT c.id, c.source, c.section, d.doc_type, d.priority, d.updated, n.distance, c.text
            FROM nearest n
            JOIN chunks c USING (id)
            JOIN documents d USING (source)
            ORDER BY n.distance
            """,
            {"q": np.asarray(query_vector, dtype=np.float32), "k": top_k},
        )
        return [
            {
                "id": id_, "source": source, "section": section, "doc_type": doc_type, "priority": priority,
                "updated": updated, "score": round(1 - distance, 4), "text": text,
            }
            for id_, source, section, doc_type, priority, updated, distance, text in rows
        ]

    def keyword_ranking(self, query: str, backend: str, limit: int) -> list[str]:
        """Chunk ids matching any of the query's terms, best first (full-text search, ts_rank).

        Terms are OR-ed, like BM25: a question rarely has every one of its
        words in the passage that answers it.
        """
        from psycopg import sql

        config, column = KEYWORD_SEARCH[backend]
        terms = tokenize(query) if backend == "pg_fts_stems" else re.findall(r"\w+", turkish_lower(query))
        if not terms:
            return []
        self._ready()
        rows = self._query(
            sql.SQL(
                """
                SELECT id
                FROM chunks, to_tsquery(%(config)s::regconfig, %(terms)s) AS query
                WHERE {column} @@ query
                ORDER BY ts_rank({column}, query, %(normalization)s) DESC, id
                LIMIT %(limit)s
                """
            ).format(column=sql.Identifier(column)),
            {"config": config, "terms": " | ".join(terms), "normalization": TS_RANK_NORMALIZATION, "limit": limit},
        )
        return [row[0] for row in rows]


def _jsonb(value):
    from psycopg.types.json import Jsonb

    return Jsonb(value)
