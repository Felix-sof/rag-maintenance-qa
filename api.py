"""
HTTP API for the knowledge-base assistant.

    uvicorn api:app --reload

POST /ask      question (+ optional history) -> cited answer
GET  /search   retrieval only, no LLM call - inspect what the model would see,
               optionally comparing retrieval modes (?mode=vector|bm25|hybrid|hybrid_rerank)
GET  /health   index stats
"""

from typing import Literal, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

import rag
import retriever
from generator import MODEL
from vector_store import MAX_TOP_K, get_store

app = FastAPI(
    title="Maintenance RAG API",
    description="Question answering over a milling-machine maintenance knowledge base, with cited sources.",
)


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["HDF arızası hangi koşullarda oluşur?"])
    history: list[Turn] = Field(default_factory=list, description="Previous turns, oldest first, for follow-ups.")
    top_k: int = Field(rag.DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


_retrievers_by_mode: dict[str, retriever.Retriever] = {}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL,
        "retrieval_mode": retriever.get_retriever().mode,
        "index": get_store().stats(),
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
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


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return rag.answer(req.question, history=[t.model_dump() for t in req.history], top_k=req.top_k)
