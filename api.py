"""
HTTP API for the knowledge-base assistant.

    uvicorn api:app --reload

POST /ask      question (+ optional history) -> cited answer
GET  /search   retrieval only, no LLM call - for inspecting what the model would see
GET  /health   index stats
"""

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

import rag
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL, "index": get_store().stats()}


@app.get("/search")
def search(q: str = Query(..., min_length=1), top_k: int = Query(rag.DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)) -> dict:
    return {"query": q, "results": get_store().search(q, top_k=top_k)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return rag.answer(req.question, history=[t.model_dump() for t in req.history], top_k=req.top_k)
