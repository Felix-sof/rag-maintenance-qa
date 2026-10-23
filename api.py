"""
HTTP API for the knowledge-base assistant.

    uvicorn api:app --reload

POST   /ask                 question (+ optional history) -> cited answer
GET    /search              retrieval only, no LLM call - inspect what the model would see,
                            optionally comparing retrieval modes (?mode=vector|bm25|hybrid|hybrid_rerank)
GET    /documents           indexed documents with chunk counts, plus any that couldn't be read
POST   /documents           upload a PDF / Markdown / text document (multipart) and index it
DELETE /documents/{source}  delete an uploaded document (uploads/ only) and re-index
GET    /health              index stats
GET    /stats               query-log analytics: daily volume, refusal rate, latency, top documents
                            (needs DATABASE_URL; the SQL is in analytics/)

For a public deployment:
    READ_ONLY=1     uploads and deletes are refused; the documents are those the
                    deployment indexed (cloud disks don't keep uploaded files anyway)
    WARM_UP=1       load the models before accepting traffic, so the first user
                    doesn't wait for them (see lifespan below)
    /ask is rate limited per client and per day, and the daily cap is also
    checked against the query log, so it survives a restart (rate_limit.py).
    The other endpoints have a looser per-client limit; inputs are capped in size.
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

import analytics
import copilot_tools
import query_log
import rag
import retriever
from generator import MODEL
from rate_limit import PER_DAY, RateLimiter
from vector_store import MAX_TOP_K, get_store


def copilot_on() -> bool:
    """COPILOT=1 (the default): /ask may call the failure-rule and dataset tools (copilot_tools.py)."""
    return os.environ.get("COPILOT", "1") == "1"


def read_only() -> bool:
    return os.environ.get("READ_ONLY", "0") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """With WARM_UP=1, load the embedding model and the reranker (and connect to the index) at startup.

    Uvicorn starts accepting requests only after this returns, and Cloud Run
    sends traffic only once the startup probe (/health) passes - so a cold
    start costs the platform's time, not the first user's.
    """
    if os.environ.get("WARM_UP", "0") == "1":
        retriever.get_retriever().search("warm-up", top_k=1)
    yield


app = FastAPI(
    title="Maintenance RAG API",
    description="Question answering over a milling-machine maintenance knowledge base, with cited sources.",
    lifespan=lifespan,
)
limiter = RateLimiter()  # /ask: it calls the LLM
light_limiter = RateLimiter(per_minute=60, per_day=5_000)  # everything else costs CPU or a database query

MAX_QUESTION_CHARS = 500  # longer input buys an attacker CPU time in the embedder and the reranker
MAX_HISTORY_TURNS = 4  # rag.py only reads the last 4 anyway
MAX_TURN_CHARS = 2_000


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _too_many(wait: int) -> HTTPException:
    return HTTPException(status_code=429, detail="too many requests", headers={"Retry-After": str(wait)})


def rate_limited(request: Request) -> None:
    """Dependency for the endpoints that call the LLM.

    The in-memory counters reset when the instance restarts - and with
    min-instances=0 it restarts after every idle spell. So the daily cap is
    also checked against the query log in Postgres, which a restart doesn't
    clear. If the database can't be reached, the in-memory cap still holds.
    """
    allowed, wait = limiter.check(_client(request))
    if not allowed:
        raise _too_many(wait)
    if query_log.enabled():
        try:
            used = query_log.get().llm_answers_last_24h(source="api")
        except Exception:  # noqa: BLE001 - an unreachable database must not take /ask down with it
            used = 0
        if used >= PER_DAY:
            raise _too_many(3600)


def lightly_limited(request: Request) -> None:
    allowed, wait = light_limiter.check(_client(request))
    if not allowed:
        raise _too_many(wait)


def writable() -> None:
    if read_only():
        raise HTTPException(status_code=403, detail="this deployment is read-only: documents are managed by its index")


class Turn(BaseModel):
    question: str = Field(..., max_length=MAX_TURN_CHARS)
    answer: str = Field(..., max_length=MAX_TURN_CHARS)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS, examples=["HDF arızası hangi koşullarda oluşur?"])
    history: list[Turn] = Field(
        default_factory=list, max_length=MAX_HISTORY_TURNS, description="Previous turns, oldest first, for follow-ups."
    )
    top_k: int = Field(rag.DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


_retrievers_by_mode: dict[str, retriever.Retriever] = {}


_health_cache: dict = {"at": float("-inf"), "index": None}
HEALTH_CACHE_S = 30


@app.get("/health")
def health() -> dict:
    """Liveness and index stats. Not rate limited - the platform's probes call it - so the
    database part is cached: a flood of /health requests can't become a flood of queries."""
    if time.monotonic() - _health_cache["at"] > HEALTH_CACHE_S:
        _health_cache.update(at=time.monotonic(), index=get_store().stats())
    return {
        "status": "ok",
        "read_only": read_only(),
        "model": MODEL,
        "retrieval_mode": retriever.get_retriever().mode,
        "index": _health_cache["index"],
    }


@app.get("/search", dependencies=[Depends(lightly_limited)])
def search(
    q: str = Query(..., min_length=1, max_length=MAX_QUESTION_CHARS),
    top_k: int = Query(rag.DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
    mode: Optional[Literal["vector", "bm25", "hybrid", "hybrid_rerank"]] = None,
) -> dict:
    if mode is None:
        active = retriever.get_retriever()
    else:
        active = _retrievers_by_mode.get(mode)
        if active is None or active.store is not get_store():
            active = _retrievers_by_mode[mode] = retriever.Retriever(store=get_store(), mode=mode)
    return {"query": q, "mode": active.mode, "results": active.search(q, top_k=top_k)}


@app.post("/ask", dependencies=[Depends(rate_limited)])
def ask(req: AskRequest) -> dict:
    result = rag.answer(
        req.question,
        history=[t.model_dump() for t in req.history],
        top_k=req.top_k,
        tools=copilot_tools if copilot_on() else None,
    )
    query_log.record(result, source="api")
    return result


STATS_QUERIES = ("daily_volume", "refusal_rate", "latency_percentiles", "top_documents")


@app.get("/stats", dependencies=[Depends(lightly_limited)])
def stats() -> dict:
    if not query_log.enabled():
        raise HTTPException(status_code=503, detail="query logging is off: set DATABASE_URL")
    log = query_log.get()
    return {name: analytics.run(log, name) for name in STATS_QUERIES}


@app.get("/documents", dependencies=[Depends(lightly_limited)])
def list_documents() -> dict:
    store = get_store()
    return {"documents": store.chunk_counts(), "skipped": store.skipped()}


@app.post("/documents", status_code=201, dependencies=[Depends(writable)])
async def upload_document(
    file: UploadFile = File(..., description="PDF, Markdown (.md) or text (.txt) document"),
    doc_type: Literal["reference", "example"] = Form("reference"),
    priority: int = Form(2, ge=1, le=10, description="Higher wins when documents conflict"),
    updated: Optional[date] = Form(None, description="Last revision date (YYYY-MM-DD); defaults to today"),
) -> dict:
    store = get_store()
    meta = {"type": doc_type, "priority": priority, "updated": (updated or date.today()).isoformat()}
    try:
        source = store.save_upload(file.filename or "", await file.read(), meta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "source": source,
        "chunks": store.chunk_counts().get(source, 0),
        "error": store.skipped().get(source),  # set if the file couldn't be read (e.g. a corrupt PDF)
    }


@app.delete("/documents/{source:path}", dependencies=[Depends(writable)])
def delete_document(source: str) -> dict:
    try:
        get_store().delete_upload(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": source}
