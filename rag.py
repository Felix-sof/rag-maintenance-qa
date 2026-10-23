"""
The RAG pipeline: question -> retrieve -> augment -> generate -> cited answer.

    1. Condense   - a follow-up like "peki adımları neler?" is meaningless to
                    the retriever on its own; with conversation history, the
                    LLM first rewrites it as a standalone question.
    2. Retrieve   - top-k chunks via hybrid search (vectors + BM25, fused,
                    re-ranked by a cross-encoder - see retriever.py).
    3. Gate       - if even the best chunk is a weak match (reranker score, or
                    cosine in the modes without a reranker), the question is
                    off-topic: answer "not in the knowledge base" without
                    calling the LLM at all.
    4. Augment    - number the retrieved chunks [1]..[k] into a context block.
    5. Generate   - Gemini answers ONLY from that context, citing [n].
    6. Verify     - parse which [n] the answer actually cites, so callers can
                    show the sources that were used, not just the retrieved ones,
                    and check that every number in the answer is in one of them
                    (grounding.py).

    python rag.py "HDF arızası nasıl önlenir?"
"""

import json
import re
import sys
import time

from generator import GenerationError, get_generator
from grounding import unsupported_numbers
from retriever import get_retriever

DEFAULT_TOP_K = 4
MAX_HISTORY_TURNS = 4

# The off-topic gate: below these floors the best retrieved chunk is taken as
# unrelated, and the question is refused without an LLM call. Both floors were
# measured on this knowledge base with `python -m evals.gate_eval`; re-measure
# if you change a model or the documents substantially.
#
# MIN_RERANK_SCORE applies when a cross-encoder scored the hits (the default
# hybrid_rerank mode). It is a raw logit of mmarco-mMiniLMv2: in-scope
# questions scored -4.58 or higher, and -5.0 refuses none of them while
# stopping about two thirds of unrelated questions and a third of near-domain
# ones the documents don't cover ("bu ay kaç arıza oldu").
#
# MIN_SCORE is the embedding cosine, used in the modes without a reranker.
# multilingual-e5 compresses every score into a narrow band, so it is the
# weaker signal: it stops more unrelated questions, but it also refuses
# in-scope ones phrased unlike the documents ("Rulmandan ses geliyor" 0.764,
# English questions around 0.79).
#
# Neither floor separates near-domain gaps from in-scope questions - the rest
# of those are caught by the prompt rule below.
MIN_SCORE = 0.80
MIN_RERANK_SCORE = -5.0

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

TOOL_RULES = f"""
Tools - you can also call tools:
- assess_process applies the documented failure rules to readings the user gives, and
  compares them with the most similar recorded processes. query_dataset answers questions
  about the recorded AI4I processes. Call a tool when the question needs one; answer from
  the passages alone when it doesn't.
- Your tool results are numbered in the order you receive them: [T1], [T2], ... Cite them
  like passages, e.g. "HDF koşulu sağlanıyor [T1].", and cite the passage that states the
  rule too.
- Report a tool's numbers as it gives them (values, thresholds, margins, rates). Don't
  compute new ones.
- A TWF or RNF rate is the share of recorded processes that failed, not a prediction for
  this one: say so. Dataset figures come from a synthetic dataset: say that once.
- If a tool returns an error, say what is missing or wrong instead of guessing. If the
  readings are outside the dataset's range (outside_data_range), say the assessment
  extrapolates.
- Reply "{NO_ANSWER}" only if neither the passages nor a tool answers the question.
"""

# Two or more numbers with a sensor unit ("300 K", "1300 rpm", "45 Nm", "215 dk"): the
# question is about specific readings, which the tools answer even when no passage matches it.
READINGS_PATTERN = re.compile(r"\d+(?:[.,]\d+)?\s*(?:K|°C|rpm|Nm|dk|dak|min)\b", re.IGNORECASE)

CONDENSE_PROMPT = """Rewrite the follow-up question so it can be understood without the
conversation, keeping its language. Resolve pronouns and references ("peki onun
adımları?", "bunu nasıl önleriz?") using the conversation. If it is already
standalone, return it unchanged. Return only the rewritten question.

Conversation:
{history}

Follow-up question: {question}"""


def format_passage(hit: dict) -> str:
    """One passage as the model sees it: a header (source, section, type, priority, date), then the text."""
    tags = ["örnek belge"] if hit["doc_type"] == "example" else []
    tags.append(f"öncelik {hit.get('priority', 1)}")
    if hit.get("updated"):
        tags.append(f"güncelleme {hit['updated']}")
    return f"{hit['source']} > {hit['section']} ({', '.join(tags)})\n{hit['text']}"


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{n}] {format_passage(hit)}" for n, hit in enumerate(hits, start=1))


def extract_citations(answer: str, n_sources: int) -> list[int]:
    """Source numbers the answer cites, ignoring any [n] outside 1..n_sources."""
    cited = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):  # [1], [1, 3], [2,4]
        for num in re.findall(r"\d+", group):
            if 1 <= int(num) <= n_sources:
                cited.add(int(num))
    return sorted(cited)


def extract_tool_citations(answer: str, n_calls: int) -> list[int]:
    """Tool results the answer cites as [T1], [T2], ..., ignoring any outside 1..n_calls."""
    return sorted({int(n) for n in re.findall(r"\[T(\d+)\]", answer) if 1 <= int(n) <= n_calls})


def gives_readings(question: str) -> bool:
    return len(READINGS_PATTERN.findall(question)) >= 2


def is_off_topic(hits: list[dict]) -> bool:
    """True if even the best hit is too weak a match to be worth an LLM call."""
    if not hits:
        return True
    rerank_scores = [h["ranks"]["rerank"] for h in hits if (h.get("ranks") or {}).get("rerank") is not None]
    if rerank_scores:
        return max(rerank_scores) < MIN_RERANK_SCORE
    return max(h["score"] for h in hits) < MIN_SCORE


def condense_question(question: str, history: list[dict], generator) -> str:
    turns = history[-MAX_HISTORY_TURNS:]
    transcript = "\n".join(f"Kullanıcı: {t['question']}\nAsistan: {t['answer']}" for t in turns)
    rewritten = generator.generate(
        system="You rewrite follow-up questions into standalone questions.",
        prompt=CONDENSE_PROMPT.format(history=transcript, question=question),
        temperature=0.0,
    )
    return rewritten or question


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class _MeteredGenerator:
    """Wraps the generator to time its calls; it counts calls and tokens into result["usage"] itself."""

    def __init__(self, generator, result: dict):
        self.generator = generator
        self.result = result

    def generate(self, system: str, prompt: str, temperature: float = 0.2) -> str:
        start = time.perf_counter()
        try:
            return self.generator.generate(system, prompt, temperature, usage=self.result["usage"])
        finally:
            self.result["timings"]["llm_ms"] += _ms_since(start)

    def generate_with_tools(self, system: str, prompt: str, tools, call_tool) -> tuple[str, list[dict]]:
        """Tool time is LLM time here too: it is part of the same exchange with the model."""
        start = time.perf_counter()
        try:
            return self.generator.generate_with_tools(system, prompt, tools, call_tool, usage=self.result["usage"])
        finally:
            self.result["timings"]["llm_ms"] += _ms_since(start)


def answer(
    question: str, history: list[dict] = None, top_k: int = DEFAULT_TOP_K, retriever=None, generator=None, tools=None
) -> dict:
    """Answer a question from the knowledge base.

    `history` is a list of previous {"question", "answer"} turns (oldest
    first) for follow-up questions; omit it for a standalone question.

    Returns a dict with "answer", "sources" (every retrieved chunk, numbered
    as in the prompt), "cited" (the source numbers the answer actually
    uses), "search_query" (the question after condensing), "conflict" (True
    if the answer reports that its sources disagree), "unsupported_numbers"
    (numbers in the answer that none of its cited sources contain - see
    grounding.py), "error" (True only if the LLM call failed), and for the
    query log: "outcome" (answered / refused_gate / refused_llm / error),
    "timings" (total, retrieval and LLM milliseconds), "usage" (LLM calls
    and tokens), "model" and "retrieval_mode".

    With `tools` (a module or object with TOOLS and call(), such as
    copilot_tools), the model may call them before answering: "tool_calls"
    lists each call with its result, and "cited_tools" the [Tn] the answer
    cites. A question that gives sensor readings skips the off-topic gate,
    since the tools can answer it even when no passage matches.
    """
    started = time.perf_counter()
    retriever = retriever or get_retriever()
    generator = generator or get_generator()
    result = {
        "question": question,
        "search_query": question,
        "answer": "",
        "sources": [],
        "cited": [],
        "tool_calls": [],
        "cited_tools": [],
        "conflict": False,
        "unsupported_numbers": [],
        "error": False,
        "outcome": "answered",
        "model": getattr(generator, "model", None),
        "retrieval_mode": getattr(retriever, "mode", None),
        "timings": {"total_ms": 0, "retrieval_ms": None, "llm_ms": 0},
        "usage": {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0},
    }
    generator = _MeteredGenerator(generator, result)

    try:
        search_query = condense_question(question, history, generator) if history else question
        result["search_query"] = search_query

        retrieval_started = time.perf_counter()
        hits = retriever.search(search_query, top_k=top_k)
        result["timings"]["retrieval_ms"] = _ms_since(retrieval_started)
        result["sources"] = [{"n": n, **hit} for n, hit in enumerate(hits, start=1)]
        if is_off_topic(hits) and not (tools and gives_readings(search_query)):
            result["answer"] = NO_ANSWER
            result["outcome"] = "refused_gate"
            return result

        prompt = f"Context:\n\n{format_context(hits)}\n\nQuestion: {search_query}"
        if tools:
            result["answer"], result["tool_calls"] = generator.generate_with_tools(
                SYSTEM_PROMPT + TOOL_RULES, prompt, tools.TOOLS, tools.call
            )
        else:
            result["answer"] = generator.generate(SYSTEM_PROMPT, prompt)
        result["outcome"] = "refused_llm" if NO_ANSWER in result["answer"] else "answered"

        result["cited"] = extract_citations(result["answer"], len(hits))
        result["cited_tools"] = extract_tool_citations(result["answer"], len(result["tool_calls"]))
        result["conflict"] = CONFLICT_MARKER in result["answer"]
        result["unsupported_numbers"] = unsupported_numbers(
            result["answer"],
            # a tool result counts as a source whether or not the answer cites it: the model saw it
            [format_passage(hits[n - 1]) for n in result["cited"]]
            + [json.dumps(call["result"], ensure_ascii=False) for call in result["tool_calls"]],
            question=f"{question} {search_query}",
        )
        return result
    except GenerationError as exc:
        result["answer"] = f"Cevap üretilemedi: {exc}"
        result["error"] = True
        result["outcome"] = "error"
        return result
    finally:
        result["timings"]["total_ms"] = _ms_since(started)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    if not query:
        print(__doc__)
        sys.exit(0)
    out = answer(query)
    print(out["answer"], "\n")
    if out["unsupported_numbers"]:
        print(f" ! atıf yapılan kaynaklarda bulunmayan değerler: {', '.join(out['unsupported_numbers'])}\n")
    for src in out["sources"]:
        mark = "*" if src["n"] in out["cited"] else " "
        print(f" {mark}[{src['n']}] {src['score']:.3f}  {src['source']} > {src['section']}")
