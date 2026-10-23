"""Tests for query_log.py and the analytics SQL, against a real PostgreSQL.

The analytics tests seed a handful of rows whose answers can be worked out
by hand (counts, a refusal rate, an interpolated p95) and check each query
returns exactly those. Skipped unless TEST_DATABASE_URL is set, except the
ones that need no database.
"""

import logging
import os
import uuid
from decimal import Decimal

import pytest

import analytics
import query_log

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
needs_postgres = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")


def hit(n, source="failure_modes.md", section="Arızalar > HDF", score=0.85, rerank=2.0):
    return {"n": n, "id": f"{source}-{n}", "source": source, "section": section, "score": score,
            "ranks": {"vector": n, "bm25": None, "rerank": rerank}}


def result(outcome="answered", latency=100, sources=None, cited=(1,), llm_calls=1, question="HDF nedir?"):
    return {
        "question": question,
        "search_query": question,
        "answer": "...",
        "outcome": outcome,
        "sources": [hit(1), hit(2, source="safety.md", section="Kilitleme", rerank=-1.0)] if sources is None else sources,
        "cited": list(cited),
        "conflict": False,
        "unsupported_numbers": [],
        "timings": {"total_ms": latency, "retrieval_ms": 40, "llm_ms": latency - 40 if llm_calls else 0},
        "usage": {"llm_calls": llm_calls, "prompt_tokens": 900 if llm_calls else 0, "output_tokens": 100 if llm_calls else 0},
        "model": "gemini-test",
        "retrieval_mode": "hybrid_rerank",
    }


@pytest.fixture
def log():
    import psycopg
    from psycopg import sql

    schema = f"test_{uuid.uuid4().hex[:12]}"
    qlog = query_log.QueryLog(DATABASE_URL, schema=schema)
    yield qlog
    qlog.close()
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def backdate(log, log_id, days):
    log.connection().execute("UPDATE query_logs SET created_at = now() - make_interval(days => %s) WHERE id = %s", (days, log_id))


@needs_postgres
def test_a_result_becomes_a_log_row_and_one_row_per_hit(log):
    log_id = log.record(result(cited=[2]), source="api")
    row = log.fetch(f"SELECT source, outcome, top_cosine, top_rerank, prompt_tokens, llm_ms FROM query_logs WHERE id = {log_id}")[0]
    assert row["source"] == "api" and row["outcome"] == "answered"
    assert row["top_rerank"] == pytest.approx(2.0) and row["top_cosine"] == pytest.approx(0.85)
    assert (row["prompt_tokens"], row["llm_ms"]) == (900, 60)
    hits = log.fetch(f"SELECT rank, source, cited FROM query_log_hits WHERE query_log_id = {log_id} ORDER BY rank")
    assert hits == [
        {"rank": 1, "source": "failure_modes.md", "cited": False},
        {"rank": 2, "source": "safety.md", "cited": True},
    ]


@needs_postgres
def test_llm_answers_in_the_last_day_count_toward_the_daily_cap(log):
    log.record(result(), source="api")
    log.record(result(outcome="refused_gate", llm_calls=0, cited=()), source="api")  # no LLM call: free
    log.record(result(), source="ui")  # another source
    backdate(log, log.record(result(), source="api"), days=2)  # older than 24 h
    assert log.llm_answers_last_24h("api") == 1


@needs_postgres
def test_no_llm_call_means_no_token_or_llm_time(log):
    log_id = log.record(result(outcome="refused_gate", llm_calls=0, cited=()), source="ui")
    row = log.fetch(f"SELECT prompt_tokens, output_tokens, llm_ms, llm_calls FROM query_logs WHERE id = {log_id}")[0]
    assert row == {"prompt_tokens": None, "output_tokens": None, "llm_ms": None, "llm_calls": 0}


@needs_postgres
def test_an_unknown_outcome_is_rejected_by_the_database(log):
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        log.record(result(outcome="maybe"), source="api")


def test_logging_failure_never_reaches_the_caller(monkeypatch, caplog):
    import psycopg

    import db

    def unreachable(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@db.invalid/nothing")
    monkeypatch.setenv("QUERY_LOG", "on")
    monkeypatch.setattr(query_log, "_log", None)
    monkeypatch.setattr(db, "connect", unreachable)
    monkeypatch.setattr(query_log, "_paused_until", 0.0)
    with caplog.at_level(logging.WARNING, logger="query_log"):
        query_log.record(result(), source="api")  # must not raise
    assert "query not logged" in caplog.text

    attempts = []
    monkeypatch.setattr(db, "connect", lambda *a, **k: attempts.append(1) or unreachable())
    query_log.record(result(), source="api")
    assert attempts == []  # paused: the next request doesn't wait on the database again


def test_logging_is_off_without_a_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("QUERY_LOG", "on")
    assert query_log.enabled() is False


@pytest.fixture
def seeded(log):
    """Two days of traffic whose statistics are easy to work out by hand."""
    two_days_ago = [
        result("answered", 100, cited=[1]),
        result("answered", 300, cited=[1]),
        result("refused_gate", 50, llm_calls=0, cited=(), question="Pizza?"),
    ]
    today = [
        result("refused_llm", 200, cited=(), question="Hidrolik pres?"),
        result("error", 1000, cited=()),
        result("answered", 500, cited=[1, 2]),
    ]
    for item in two_days_ago:
        backdate(log, log.record(item, source="replay"), days=2)
    for item in today:
        log.record(item, source="api")
    return log


@needs_postgres
class TestAnalytics:
    def test_every_query_runs(self, seeded):
        assert set(analytics.queries()) == {
            "daily_volume", "refusal_rate", "latency_percentiles", "top_documents", "hard_questions", "weekly_trend",
            "tool_usage",
        }
        for name in analytics.queries():
            assert analytics.run(seeded, name), name

    def test_daily_volume(self, seeded):
        days = analytics.run(seeded, "daily_volume")
        assert [(d["questions"], d["answered"], d["refused"], d["errors"]) for d in days] == [(3, 2, 1, 0), (3, 1, 1, 1)]

    def test_refusal_rate_leaves_errors_out(self, seeded):
        total = analytics.run(seeded, "refusal_rate")[-1]
        assert total["source"] == "all"
        assert total["questions"] == 5  # the error is not a verdict
        assert total["refusal_rate"] == Decimal("0.400")  # 2 of 5
        assert (total["refused_by_gate"], total["refused_by_llm"], total["gate_share"]) == (1, 1, Decimal("0.500"))

    def test_latency_percentiles_interpolate(self, seeded):
        answered = next(r for r in analytics.run(seeded, "latency_percentiles") if r["outcome"] == "answered")
        # latencies 100, 300, 500: p50 = 300; p95 sits 90% of the way from 300 to 500
        assert (answered["p50_ms"], answered["p95_ms"]) == (300, 480)
        assert answered["avg_tokens"] == 1000

    def test_top_documents_count_only_passages_the_llm_read(self, seeded):
        docs = {d["source"]: d for d in analytics.run(seeded, "top_documents")}
        # 3 answered + 1 refused by the LLM read both passages; the gate refusal and the error are left out
        assert docs["failure_modes.md"]["retrieved"] == 4
        assert docs["failure_modes.md"]["cited"] == 3
        assert docs["safety.md"]["cite_rate"] == Decimal("0.250")

    def test_hard_questions_include_llm_refusals(self, seeded):
        questions = [r["question"] for r in analytics.run(seeded, "hard_questions")]
        assert "Hidrolik pres?" in questions
        assert "Pizza?" not in questions  # a gate refusal is off-topic, not hard

    def test_tool_usage_counts_each_tool_and_the_plain_answers(self, seeded):
        with_tools = result()
        with_tools["tool_calls"] = [{"name": "assess_process"}, {"name": "query_dataset"}]
        with_tools["usage"]["llm_calls"] = 2
        seeded.record(with_tools, source="api")
        usage = {r["tool"]: r for r in analytics.run(seeded, "tool_usage")}
        assert usage["assess_process"]["questions"] == usage["query_dataset"]["questions"] == 1
        assert usage["(none)"]["questions"] == 5  # the seeded answers, refusals and error - not the gate refusal
        assert float(usage["assess_process"]["avg_llm_calls"]) == 2

    def test_weekly_trend_has_no_gaps_and_a_running_total(self, seeded):
        backdate(seeded, seeded.record(result(), source="api"), days=21)  # three weeks back
        weeks = analytics.run(seeded, "weekly_trend")
        assert len(weeks) >= 4  # the empty weeks in between are filled in
        assert all((b["week"] - a["week"]).days == 7 for a, b in zip(weeks, weeks[1:]))
        assert weeks[-1]["running_total"] == 7
        assert 0 in [w["questions"] for w in weeks]


@needs_postgres
def test_stats_endpoint(seeded, monkeypatch):
    from fastapi.testclient import TestClient

    import api

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("QUERY_LOG", "on")
    monkeypatch.setattr(query_log, "_log", seeded)
    body = TestClient(api.app).get("/stats").json()
    assert set(body) == set(api.STATS_QUERIES)
    assert body["refusal_rate"][-1]["questions"] == 5


@needs_postgres
def test_analytics_page_renders(seeded, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("QUERY_LOG", "on")
    monkeypatch.setattr(query_log, "_log", seeded)
    page = AppTest.from_file("../pages/analytics.py", default_timeout=30).run()
    assert not page.exception
    assert [m.label for m in page.metric][:2] == ["Soru (son 30 gün)", "Reddetme oranı"]
    assert page.metric[1].value == "40%"


def test_stats_endpoint_without_a_database():
    from fastapi.testclient import TestClient

    import api

    assert TestClient(api.app).get("/stats").status_code == 503


@needs_postgres
def test_the_tools_a_question_used_are_logged(log):
    with_tools = result()
    with_tools["tool_calls"] = [{"name": "assess_process", "args": {}, "result": {}}]
    log_id = log.record(with_tools, source="api")
    assert log.fetch("SELECT tools FROM query_logs WHERE id = %s", (log_id,))[0]["tools"] == ["assess_process"]
