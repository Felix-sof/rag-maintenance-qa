"""
End-to-end answer eval: runs rag.answer() against the live Gemini API and
grades each answer programmatically (no LLM judge).

    python -m evals.answer_eval
    python -m evals.answer_eval --only hdf_conditions,out_of_scope_near   # re-run some, merge into results
    python -m evals.answer_eval --retry-infra                              # re-run only infra_error cases

Checks per case:
    expect_sources    at least one CITED source must match one of these
                      "source > section" substrings - retrieving the right
                      section isn't enough, the answer has to use it
    expect_terms      list of any-of groups; every group must appear in the
                      answer (case-insensitive) - e.g. the 8.6 K / 1380 rpm
                      HDF thresholds, written either "8,6" or "8.6"
    expect_no_answer  the answer must be the NO_ANSWER refusal (off-topic or
                      not covered by the documents)
    history           optional earlier turns, to test follow-up condensing
    extra_docs        {file name: content} added to a temporary copy of the
                      knowledge base for this case only (e.g. a conflicting
                      document); knowledge/ itself is never modified
    expect_conflict   the answer must flag the conflict (CONFLICT_MARKER).
                      Every other case must NOT flag one - a false alarm on
                      agreeing sources is a failure too.

Any answer that isn't a refusal must cite at least one source.
A case whose LLM call failed (quota, network) is reported as infra_error and
not graded - that's infrastructure noise, not a signal about the pipeline.

Results go to evals/results/answer_eval.json.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import rag
from generator import MODEL
from retriever import Retriever, get_retriever
from vector_store import DOCS_DIR, VectorStore, get_store

RESULTS_PATH = Path(__file__).parent / "results" / "answer_eval.json"
PAUSE_SECONDS = 5  # stay well under the free-tier requests-per-minute limit

CASES = [
    # --- facts with exact thresholds (reference docs) ---
    {
        "id": "hdf_conditions",
        "question": "HDF arızası hangi koşullarda oluşur?",
        "expect_sources": ["> HDF"],
        "expect_terms": [["8,6", "8.6"], ["1380", "1.380"]],
    },
    {
        "id": "pwf_power_band",
        "question": "Güç arızası hangi güç değerlerinde oluşur?",
        "expect_sources": ["> PWF"],
        "expect_terms": [["3500", "3.500", "3 500"], ["9000", "9.000", "9 000"]],
    },
    {
        "id": "osf_limit_l",
        "question": "L tipi ürünlerde aşırı zorlanma arızası sınırı nedir?",
        "expect_sources": ["> OSF"],
        "expect_terms": [["11.000", "11000", "11,000", "11 000"]],
    },
    {
        "id": "osf_limit_h_english",
        "question": "What is the overstrain failure limit for H type products?",
        "expect_sources": ["> OSF"],
        "expect_terms": [["13.000", "13000", "13,000", "13 000"]],
    },
    {
        "id": "twf_wear_window",
        "question": "Takım aşınması arızası hangi aşınma süresi aralığında olur?",
        "expect_sources": ["> TWF"],
        "expect_terms": [["200"], ["240"]],
    },
    {
        "id": "twf_label_nuance",
        "question": "TWF etiketi her zaman takımın kırıldığını mı gösterir?",
        "expect_sources": ["> TWF"],
        "expect_terms": [["değiştir"]],
    },
    # --- procedures / troubleshooting / safety (example docs) ---
    {
        "id": "tool_change_steps",
        "question": "Kesici takım nasıl değiştirilir, adımlar neler?",
        "expect_sources": ["Kesici Takım Değişimi"],
        "expect_terms": [["ofset", "offset"]],
    },
    {
        "id": "high_torque_causes",
        "question": "Tork değerleri sürekli yüksek çıkıyor, olası nedenleri neler?",
        "expect_sources": ["Tork Değerleri Yüksek"],
        "expect_terms": [["aşın", "kör"]],
    },
    {
        "id": "lockout_tagout",
        "question": "Bakıma başlamadan önce enerji izolasyonu nasıl yapılır?",
        "expect_sources": ["Enerji İzolasyonu"],
        "expect_terms": [["kilit", "kilid"]],  # Turkish t->d mutation: "kilidinizi"
    },
    {
        "id": "gloves_near_spindle",
        "question": "Dönen iş mili yakınında eldiven kullanılır mı?",
        "expect_sources": ["Kişisel Koruyucu"],
        "expect_terms": [["dolan"]],
    },
    # --- follow-up: needs condensing to retrieve the right section ---
    {
        "id": "followup_cooling_steps",
        "history": [
            {
                "question": "Soğutma sistemi ne zaman kontrol edilmeli?",
                "answer": "Proses ve hava sıcaklığı farkı azaldığında veya HDF arızaları arttığında [1].",
            }
        ],
        "question": "Peki kontrol adımları neler?",
        "expect_sources": ["Soğutma Sistemi Kontrolü"],
        "expect_terms": [["nozul"]],
    },
    # --- must refuse ---
    {
        "id": "out_of_scope_near",  # same domain, not in the docs: the prompt rule must catch it
        "question": "Hidrolik presin yağ değişimi nasıl yapılır?",
        "expect_no_answer": True,
    },
    {
        "id": "out_of_scope_far",  # unrelated: the MIN_SCORE gate should catch it
        "question": "En iyi pizza tarifi nedir?",
        "expect_no_answer": True,
    },
    {
        "id": "out_of_scope_live_data",  # the docs describe failures, they don't count them
        "question": "Bu ay fabrikada kaç arıza oldu?",
        "expect_no_answer": True,
    },
    # --- conflicting documents: must surface both values, not silently pick one ---
    {
        "id": "conflict_higher_priority_wins",
        "extra_docs": {
            "hdf_saha_notu.md": (
                "---\ntitle: HDF Saha Notu\ntype: reference\npriority: 3\nupdated: 2026-09-15\n---\n\n"
                "# HDF Saha Notu\n\n## HDF Tetiklenme Koşulu (Revize)\n\n"
                "Saha ölçümlerine göre ısı dağıtım arızası (HDF), hava ile proses sıcaklığı farkı "
                "8,6 K'nin altındayken dönüş hızı 1500 rpm'nin altına düştüğünde oluşur.\n"
            )
        },
        "question": "HDF arızası hangi dönüş hızının altında oluşur?",
        "expect_conflict": True,
        "expect_terms": [["1380", "1.380"], ["1500", "1.500"]],
    },
    {
        "id": "conflict_equal_priority_newer_wins",
        "extra_docs": {
            "sogutma_eski_talimat.md": (
                "---\ntitle: Eski Soğutma Talimatı\ntype: example\npriority: 1\nupdated: 2024-01-10\n---\n\n"
                "# Eski Soğutma Talimatı\n\n## Soğutma Sıvısı Kontrol Sıklığı\n\n"
                "Soğutma sıvısı seviyesi ayda bir kontrol edilir.\n"
            )
        },
        "question": "Soğutma sıvısı seviyesi ne sıklıkla kontrol edilmeli?",
        "expect_conflict": True,
        "expect_terms": [["vardiya"], ["ayda"]],
    },
]


def grade(case: dict, out: dict) -> tuple[str, list[str]]:
    if out["error"]:
        return "infra_error", [out["answer"][:200]]

    answer = out["answer"]
    refused = rag.NO_ANSWER in answer
    notes = []

    if case.get("expect_no_answer"):
        if not refused:
            notes.append("expected a refusal, got an answer")
        return ("fail" if notes else "pass"), notes

    if refused:
        return "fail", ["refused a question the knowledge base covers"]

    if not out["cited"]:
        notes.append("answer cites no source")

    cited_labels = [f"{s['source']} > {s['section']}" for s in out["sources"] if s["n"] in out["cited"]]
    expected = case.get("expect_sources", [])
    if expected and not any(e in label for e in expected for label in cited_labels):
        notes.append(f"expected a cited source matching {expected}, cited {cited_labels}")

    lowered = answer.lower()
    for group in case.get("expect_terms", []):
        if not any(term.lower() in lowered for term in group):
            notes.append(f"answer is missing any of {group}")

    if case.get("expect_conflict") and not out["conflict"]:
        notes.append("sources conflict, but the answer did not flag it")
    if not case.get("expect_conflict") and out["conflict"]:
        notes.append("flagged a conflict between sources that agree")

    return ("fail" if notes else "pass"), notes


def retriever_with_extra_docs(extra_docs: dict, workdir: Path) -> Retriever:
    """A retriever over a temporary copy of knowledge/ plus extra documents."""
    docs = workdir / "knowledge"
    shutil.copytree(DOCS_DIR, docs)
    for name, content in extra_docs.items():
        (docs / name).write_text(content, encoding="utf-8")
    store = VectorStore(docs_dir=docs, index_dir=workdir / "index", embedder=get_store().embedder)
    default = get_retriever()
    return Retriever(store=store, mode=default.mode, reranker=default.reranker)


def run_case(case: dict) -> dict:
    if not case.get("extra_docs"):
        return rag.answer(case["question"], history=case.get("history"))
    workdir = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    try:
        retriever = retriever_with_extra_docs(case["extra_docs"], workdir)
        return rag.answer(case["question"], history=case.get("history"), retriever=retriever)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # chromadb may still hold the files open on Windows


def run(only: set = None) -> list[dict]:
    cases = [c for c in CASES if not only or c["id"] in only]
    rows = []
    for i, case in enumerate(cases):
        if i:
            time.sleep(PAUSE_SECONDS)
        start = time.monotonic()
        out = run_case(case)
        status, notes = grade(case, out)
        rows.append(
            {
                "id": case["id"],
                "status": status,
                "notes": notes,
                "question": case["question"],
                "search_query": out["search_query"],
                "answer": out["answer"],
                "cited": out["cited"],
                "conflict": out["conflict"],
                "top_score": max((s["score"] for s in out["sources"]), default=None),
                "elapsed_s": round(time.monotonic() - start, 1),
                "model": MODEL,
            }
        )
        print(f"[{i + 1}/{len(cases)}] {status.upper():11s} {case['id']}")
        for note in notes:
            print(f"              - {note}")
    return rows


if __name__ == "__main__":
    only = None
    if "--retry-infra" in sys.argv:
        previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
        only = {r["id"] for r in previous if r["status"] == "infra_error"}
        if not only:
            print("No infra_error cases in the last run.")
            sys.exit(0)
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))

    rows = run(only)
    if only and RESULTS_PATH.exists():  # merge a partial re-run into the previous full run
        by_id = {r["id"]: r for r in json.loads(RESULTS_PATH.read_text(encoding="utf-8"))}
        by_id.update({r["id"]: r for r in rows})
        rows = [by_id[c["id"]] for c in CASES if c["id"] in by_id]
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    graded = [r for r in rows if r["status"] != "infra_error"]
    passed = sum(r["status"] == "pass" for r in graded)
    print("\n" + "=" * 60)
    print(f"Model: {MODEL}")
    if graded:
        print(f"Passed: {passed}/{len(graded)} ({passed / len(graded):.0%})")
    infra = len(rows) - len(graded)
    if infra:
        print(f"Infra errors (not graded, re-run with --only): {infra}")
    print(f"Results: {RESULTS_PATH}")
