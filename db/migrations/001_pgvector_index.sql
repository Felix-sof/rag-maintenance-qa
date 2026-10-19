-- 001: the knowledge-base index for VECTOR_BACKEND=pgvector.
--
-- documents is one row per file; chunks is one row per retrievable section,
-- with its embedding and two full-text representations. index_state records
-- which version of the documents the rows were built from (a fingerprint of
-- the files, the embedding model and the chunk size), so the app can tell
-- when the index is stale - the same rule the ChromaDB backend follows.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    source      text PRIMARY KEY,               -- path under knowledge/, e.g. 'uploads/kilavuz.pdf'
    title       text NOT NULL,
    doc_type    text NOT NULL,                  -- 'reference' | 'example' | 'unknown'
    priority    integer NOT NULL,               -- higher wins when documents conflict
    updated     text NOT NULL DEFAULT '',       -- revision date as written in the metadata; '' if unknown
    metadata    jsonb NOT NULL DEFAULT '{}',    -- the raw front matter / sidecar, whatever keys it has
    indexed_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id          text PRIMARY KEY,               -- stable hash of source + section + position (chunking.py)
    source      text NOT NULL REFERENCES documents (source) ON DELETE CASCADE,
    position    integer NOT NULL,               -- order within the document
    section     text NOT NULL,                  -- heading path, e.g. 'Arıza Modları > HDF'
    text        text NOT NULL,
    -- 384 = multilingual-e5-small. A model with another size needs a migration
    -- that changes this column and rebuilds the index below.
    embedding   vector(384) NOT NULL,
    -- two keyword representations, compared in evals/retrieval_eval.py:
    tsv_turkish tsvector NOT NULL,              -- Postgres's Snowball stemmer for Turkish
    tsv_stems   tsvector NOT NULL               -- the BM25 retriever's own 5-character stems
);

-- Approximate nearest-neighbour search on cosine distance (the <=> operator).
-- At this knowledge base's size a sequential scan is just as fast; the index
-- is what keeps the same query fast at hundreds of thousands of chunks.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- Inverted indexes for the full-text match operator (@@).
CREATE INDEX chunks_tsv_turkish_gin ON chunks USING gin (tsv_turkish);
CREATE INDEX chunks_tsv_stems_gin ON chunks USING gin (tsv_stems);

CREATE INDEX chunks_source_idx ON chunks (source);

CREATE TABLE index_state (
    id          integer PRIMARY KEY CHECK (id = 1),     -- a single row
    fingerprint text NOT NULL,
    skipped     jsonb NOT NULL DEFAULT '{}',            -- files the last build couldn't read, with the reason
    built_at    timestamptz NOT NULL DEFAULT now()
);
