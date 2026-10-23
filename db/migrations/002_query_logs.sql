-- 002: one row per question asked, for the queries in analytics/.
--
-- query_logs holds what happened to a question; query_log_hits holds the
-- chunks retrieved for it, one row each, so "which documents are retrieved
-- most" is a GROUP BY rather than unpacking an array in every query.

CREATE TABLE query_logs (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at          timestamptz NOT NULL DEFAULT now(),
    source              text NOT NULL,          -- who asked: 'api', 'ui', or 'replay' (analytics/replay.py)
    question            text NOT NULL,
    search_query        text NOT NULL,          -- the follow-up rewritten as a standalone question; else = question
    -- One status column instead of refused/error flags: it can't hold an
    -- impossible combination such as refused and failed at once.
    outcome             text NOT NULL CHECK (outcome IN (
                            'answered',         -- the LLM answered from the documents
                            'refused_gate',     -- off-topic, refused by the score gate before any LLM call
                            'refused_llm',      -- the LLM found no answer in the retrieved passages
                            'error'             -- the LLM call failed (quota, network)
                        )),
    conflict            boolean NOT NULL DEFAULT false,
    unsupported_numbers text[] NOT NULL DEFAULT '{}',
    top_cosine          real,                   -- best embedding similarity among the hits
    top_rerank          real,                   -- best cross-encoder score; NULL without a reranker
    latency_ms          integer NOT NULL,       -- the whole pipeline
    retrieval_ms        integer,
    llm_ms              integer,
    llm_calls           smallint NOT NULL DEFAULT 0,
    prompt_tokens       integer,                -- NULL when no LLM call was made
    output_tokens       integer,
    model               text,
    retrieval_mode      text
);

CREATE INDEX query_logs_created_at_idx ON query_logs (created_at);

CREATE TABLE query_log_hits (
    query_log_id  bigint NOT NULL REFERENCES query_logs (id) ON DELETE CASCADE,
    rank          smallint NOT NULL,            -- 1 = [1] in the prompt
    -- No foreign key to chunks: a rebuild replaces the chunks, and the log
    -- must outlive it. The source and section are copied for the same reason.
    chunk_id      text NOT NULL,
    source        text NOT NULL,
    section       text NOT NULL,
    cosine        real NOT NULL,
    rerank        real,
    cited         boolean NOT NULL,             -- the answer cites this passage
    PRIMARY KEY (query_log_id, rank)
);

CREATE INDEX query_log_hits_source_idx ON query_log_hits (source);
