"""
Put the gate eval's questions through the real pipeline, logged as source='replay'.

The analytics need traffic and this project has no users yet. Its own eval
questions are the honest stand-in: 130 of them, in scope, near-domain and
off-topic (evals/gate_eval.py), in a fixed shuffled order. source='replay'
keeps them apart from real 'api' and 'ui' traffic in every query.

    python -m analytics.replay --limit 30       # needs GEMINI_API_KEY and DATABASE_URL

Resumable: a question already logged with a verdict is skipped, so the run
can continue once the Gemini free tier's daily quota resets. It stops by
itself when the quota runs out. The models are loaded before the first
question, so the logged latencies are a warm instance's, not a cold start's.
"""

import argparse
import random
import sys
import time

import query_log
import rag
from evals.gate_eval import KINDS, SETS
from retriever import get_retriever


def questions() -> list[str]:
    every = [question for kinds in SETS.values() for kind in KINDS for question in kinds[kind]]
    random.Random(0).shuffle(every)  # mix the kinds; the same order on every run
    return every


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="at most this many questions in this run")
    parser.add_argument("--pause", type=float, default=5.0, help="seconds between LLM calls (rate limit)")
    args = parser.parse_args()

    if not query_log.enabled():
        sys.exit("Set DATABASE_URL (and leave QUERY_LOG on): the replay is only useful logged.")
    log = query_log.get()
    done = {
        row["question"]
        for row in log.fetch("SELECT DISTINCT question FROM query_logs WHERE source = 'replay' AND outcome <> 'error'")
    }
    todo = [q for q in questions() if q not in done][: args.limit]
    print(f"{len(done)} questions already logged, {len(todo)} to go")

    get_retriever().search("warm-up", top_k=1)  # load the embedding model and the reranker first
    for i, question in enumerate(todo, start=1):
        result = rag.answer(question)
        log.record(result, "replay")
        print(f"[{i}/{len(todo)}] {result['outcome']:12s} {result['timings']['total_ms']:6d} ms  {question}")
        if result["outcome"] == "error" and "quota exhausted" in result["answer"]:
            print("The Gemini daily quota is used up; run this again after it resets.")
            break
        if result["usage"]["llm_calls"]:
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
