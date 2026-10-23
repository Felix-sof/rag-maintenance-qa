"""
Query log: every question the assistant handles becomes a row in query_logs,
plus one row per retrieved chunk in query_log_hits (db/migrations/002). The
queries in analytics/ read them.

On when DATABASE_URL is set, whichever vector backend is in use; QUERY_LOG=off
turns it off. Logging is best-effort: a failure is logged as a warning and
swallowed, because analytics must never cost a user their answer.

The row is written before the response goes out, not in a background task.
On Cloud Run an instance only gets CPU while it is handling a request, so work
left for after the response may stall until the next request, or never run.
One insert is a few milliseconds next to an LLM call of seconds.
"""

import logging
import os
import threading
import time

from dotenv import load_dotenv

import db

load_dotenv()

logger = logging.getLogger(__name__)

INSERT_LOG = """
    INSERT INTO query_logs (
        source, question, search_query, outcome, conflict, unsupported_numbers,
        top_cosine, top_rerank, latency_ms, retrieval_ms, llm_ms, llm_calls,
        prompt_tokens, output_tokens, model, retrieval_mode
    ) VALUES (
        %(source)s, %(question)s, %(search_query)s, %(outcome)s, %(conflict)s, %(unsupported_numbers)s,
        %(top_cosine)s, %(top_rerank)s, %(latency_ms)s, %(retrieval_ms)s, %(llm_ms)s, %(llm_calls)s,
        %(prompt_tokens)s, %(output_tokens)s, %(model)s, %(retrieval_mode)s
    )
    RETURNING id
"""


def enabled() -> bool:
    return bool(os.environ.get("DATABASE_URL")) and os.environ.get("QUERY_LOG", "on").lower() != "off"


class QueryLog:
    def __init__(self, database_url: str, schema: str = None):
        self.database_url = database_url
        self.schema = schema
        self._conn = None
        # FastAPI runs requests on a thread pool: one transaction at a time on
        # the shared connection, or two requests' rows would share a transaction
        self._lock = threading.Lock()

    def connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = db.connect(self.database_url, schema=self.schema)
        return self._conn

    def record(self, result: dict, source: str) -> int:
        """Store one rag.answer() result; returns the new query_logs id."""
        import psycopg

        sources = result.get("sources", [])
        rerank_scores = [s["ranks"]["rerank"] for s in sources if (s.get("ranks") or {}).get("rerank") is not None]
        usage, timings = result.get("usage", {}), result.get("timings", {})
        called_llm = usage.get("llm_calls", 0) > 0
        row = {
            "source": source,
            "question": result["question"],
            "search_query": result.get("search_query") or result["question"],
            "outcome": result["outcome"],
            "conflict": result.get("conflict", False),
            "unsupported_numbers": result.get("unsupported_numbers", []),
            "top_cosine": max((s["score"] for s in sources), default=None),
            "top_rerank": max(rerank_scores, default=None),
            "latency_ms": timings.get("total_ms", 0),
            "retrieval_ms": timings.get("retrieval_ms"),
            "llm_ms": timings.get("llm_ms") if called_llm else None,
            "llm_calls": usage.get("llm_calls", 0),
            "prompt_tokens": usage.get("prompt_tokens") if called_llm else None,
            "output_tokens": usage.get("output_tokens") if called_llm else None,
            "model": result.get("model"),
            "retrieval_mode": result.get("retrieval_mode"),
        }
        cited = set(result.get("cited", []))
        hits = [
            (
                s["n"], s["id"], s["source"], s["section"], s["score"],
                (s.get("ranks") or {}).get("rerank"), s["n"] in cited,
            )
            for s in sources
        ]

        for attempt in (1, 2):  # once more on a fresh connection if the server dropped this one
            try:
                with self._lock:
                    conn = self.connection()
                    with conn.transaction():
                        log_id = conn.execute(INSERT_LOG, row).fetchone()[0]
                        with conn.cursor() as cur:
                            cur.executemany(
                                "INSERT INTO query_log_hits"
                                " (query_log_id, rank, chunk_id, source, section, cosine, rerank, cited)"
                                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                                [(log_id, *hit) for hit in hits],
                            )
                return log_id
            except psycopg.OperationalError:
                self._conn = None
                if attempt == 2:
                    raise

    def llm_answers_last_24h(self, source: str) -> int:
        """Questions from `source` that reached the LLM in the last 24 hours (the persistent daily cap)."""
        rows = self.fetch(
            "SELECT count(*) AS n FROM query_logs"
            " WHERE source = %s AND llm_calls > 0 AND created_at > now() - interval '24 hours'",
            (source,),
        )
        return rows[0]["n"]

    def fetch(self, query: str, params=None) -> list[dict]:
        """Rows of a read query, as dicts (the analytics queries)."""
        from psycopg.rows import dict_row

        with self._lock, self.connection().cursor(row_factory=dict_row) as cur:
            return cur.execute(query, params).fetchall()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


_log: QueryLog = None
_paused_until = 0.0  # after a failure, skip logging for a while instead of making every request wait on it
PAUSE_AFTER_FAILURE_S = 60


def get() -> QueryLog:
    """The process-wide query log (only meaningful when enabled())."""
    global _log
    if _log is None:
        _log = QueryLog(os.environ["DATABASE_URL"])
    return _log


def record(result: dict, source: str) -> None:
    """Log a rag.answer() result if logging is on. Never raises.

    If the database is unreachable, each request would otherwise wait out a
    connection timeout before answering; after a failure, logging pauses for
    PAUSE_AFTER_FAILURE_S seconds (a minimal circuit breaker).
    """
    global _paused_until
    if not enabled() or time.monotonic() < _paused_until:
        return
    try:
        get().record(result, source)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        _paused_until = time.monotonic() + PAUSE_AFTER_FAILURE_S
        logger.warning("query not logged, logging paused for %ss: %s: %s", PAUSE_AFTER_FAILURE_S, type(exc).__name__, exc)
