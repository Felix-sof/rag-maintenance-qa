"""
The RAG pipeline: question -> retrieve -> augment -> generate -> cited answer.

    1. Condense   - a follow-up like "peki adımları neler?" is meaningless to
                    the retriever on its own; with conversation history, the
                    LLM first rewrites it as a standalone question.
    2. Retrieve   - top-k chunks via hybrid search (vectors + BM25, fused,
                    re-ranked by a cross-encoder - see retriever.py).
    3. Gate       - if even the best chunk scores below MIN_SCORE, the
                    question is off-topic: answer "not in the knowledge base"
                    without calling the LLM at all.
    4. Augment    - number the retrieved chunks [1]..[k] into a context block.
    5. Generate   - Gemini answers ONLY from that context, citing [n].
    6. Verify     - parse which [n] the answer actually cites, so callers can
                    show the sources that were used, not just the retrieved ones.

    python rag.py "HDF arızası nasıl önlenir?"
"""

import re
import sys

from generator import GenerationError, get_generator
from retriever import get_retriever

DEFAULT_TOP_K = 4
MAX_HISTORY_TURNS = 4

# Cosine-similarity floor for the best retrieved chunk (always the embedding
# cosine, whichever retrieval mode ranked the chunks). multilingual-e5 scores
# are compressed into a narrow band, so this only catches clearly off-topic
# questions. Measured with intfloat/multilingual-e5-small on this knowledge
# base: in-scope questions scored 0.838-0.896, clearly unrelated ones
# ("pizza tarifi", "Türkiye'nin başkenti", "Python'da liste") 0.738-0.799.
# Near-domain questions the docs don't cover ("hidrolik pres yağ değişimi",
# 0.832) overlap with in-scope ones and can't be separated by a threshold -
# those are caught by the prompt rule below instead. Re-measure if you
# change the embedding model or the documents substantially.
MIN_SCORE = 0.80

NO_ANSWER = "Bilgi bankasında bu sorunun cevabı yok."
CONFLICT_MARKER = "⚠️ Çelişki:"

SYSTEM_PROMPT = f"""You are a maintenance assistant for a milling machine. You answer
questions using ONLY the numbered context passages from the plant's maintenance
knowledge base that come with each question.

Rules:
- Use only facts stated in the context. Do not add steps, thresholds, causes or safety
  rules from your own knowledge, even if you are confident they are true.
- After every sentence that uses the context, cite the passage(s) it came from as [n],
  e.g. "HDF düşük devirde oluşur [2]." Only cite numbers that appear in the context.
- If the context does not contain the answer, reply with exactly this sentence and
  nothing else: "{NO_ANSWER}" If it answers only part of the question, answer that
  part and say clearly which part is not covered.
- If passages disagree about the same fact (different thresholds, values or steps), do
  NOT silently pick one. Start the answer with a line beginning "{CONFLICT_MARKER}" that
  gives each version with its citation. Then say which one applies: the passage with
  the higher öncelik wins; if öncelik is equal, the more recent güncelleme date wins; if
  that still doesn't decide it, say it can't be decided from the documents and should
  be confirmed with the responsible engineer. Do not use the marker when passages merely
  add different, compatible details.
- Passages marked (örnek belge) are illustrative procedures, not a real manufacturer's
  manual. When you give steps from one, add one short note that the plant's own manual
  takes precedence.
- Answer in the language of the question. Be concise; use a numbered list for
  procedures.
"""

CONDENSE_PROMPT = """Rewrite the follow-up question so it can be understood without the
conversation, keeping its language. Resolve pronouns and references ("peki onun
adımları?", "bunu nasıl önleriz?") using the conversation. If it is already
standalone, return it unchanged. Return only the rewritten question.

Conversation:
{history}

Follow-up question: {question}"""


def format_context(hits: list[dict]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        tags = ["örnek belge"] if hit["doc_type"] == "example" else []
        tags.append(f"öncelik {hit.get('priority', 1)}")
        if hit.get("updated"):
            tags.append(f"güncelleme {hit['updated']}")
        blocks.append(f"[{n}] {hit['source']} > {hit['section']} ({', '.join(tags)})\n{hit['text']}")
    return "\n\n".join(blocks)


def extract_citations(answer: str, n_sources: int) -> list[int]:
    """Source numbers the answer cites, ignoring any [n] outside 1..n_sources."""
    cited = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):  # [1], [1, 3], [2,4]
        for num in re.findall(r"\d+", group):
            if 1 <= int(num) <= n_sources:
                cited.add(int(num))
    return sorted(cited)


def condense_question(question: str, history: list[dict], generator) -> str:
    turns = history[-MAX_HISTORY_TURNS:]
    transcript = "\n".join(f"Kullanıcı: {t['question']}\nAsistan: {t['answer']}" for t in turns)
    rewritten = generator.generate(
        system="You rewrite follow-up questions into standalone questions.",
        prompt=CONDENSE_PROMPT.format(history=transcript, question=question),
        temperature=0.0,
    )
    return rewritten or question


def answer(question: str, history: list[dict] = None, top_k: int = DEFAULT_TOP_K, retriever=None, generator=None) -> dict:
    """Answer a question from the knowledge base.

    `history` is a list of previous {"question", "answer"} turns (oldest
    first) for follow-up questions; omit it for a standalone question.

    Returns a dict with "answer", "sources" (every retrieved chunk, numbered
    as in the prompt), "cited" (the source numbers the answer actually
    uses), "search_query" (the question after condensing), "conflict" (True
    if the answer reports that its sources disagree), and "error" (True
    only if the LLM call failed).
    """
    retriever = retriever or get_retriever()
    generator = generator or get_generator()
    result = {
        "question": question,
        "search_query": question,
        "answer": "",
        "sources": [],
        "cited": [],
        "conflict": False,
        "error": False,
    }

    try:
        search_query = condense_question(question, history, generator) if history else question
        result["search_query"] = search_query

        hits = retriever.search(search_query, top_k=top_k)
        result["sources"] = [{"n": n, **hit} for n, hit in enumerate(hits, start=1)]
        if not hits or max(h["score"] for h in hits) < MIN_SCORE:
            result["answer"] = NO_ANSWER
            return result

        prompt = f"Context:\n\n{format_context(hits)}\n\nQuestion: {search_query}"
        result["answer"] = generator.generate(SYSTEM_PROMPT, prompt)
    except GenerationError as exc:
        result["answer"] = f"Cevap üretilemedi: {exc}"
        result["error"] = True
        return result

    result["cited"] = extract_citations(result["answer"], len(hits))
    result["conflict"] = CONFLICT_MARKER in result["answer"]
    return result


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    if not query:
        print(__doc__)
        sys.exit(0)
    out = answer(query)
    print(out["answer"], "\n")
    for src in out["sources"]:
        mark = "*" if src["n"] in out["cited"] else " "
        print(f" {mark}[{src['n']}] {src['score']:.3f}  {src['source']} > {src['section']}")
