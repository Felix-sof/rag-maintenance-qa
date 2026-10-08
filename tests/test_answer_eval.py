"""Tests for evals/answer_eval.py's grading logic (no API calls)."""

import rag
from evals.answer_eval import grade

CASE = {"id": "x", "question": "q", "expect_sources": ["> HDF"], "expect_terms": [["8,6", "8.6"]]}
REFUSAL_CASE = {"id": "y", "question": "q", "expect_no_answer": True}
SOURCES = [
    {"n": 1, "source": "failure_modes.md", "section": "Arızalar > HDF"},
    {"n": 2, "source": "safety.md", "section": "Kilitleme"},
]


def _out(answer, cited, error=False):
    return {"answer": answer, "cited": cited, "sources": SOURCES, "error": error}


def test_pass_when_expected_source_cited_and_terms_present():
    assert grade(CASE, _out("Fark 8.6 K altında [1].", [1])) == ("pass", [])


def test_fail_when_right_source_retrieved_but_not_cited():
    status, notes = grade(CASE, _out("Fark 8,6 K altında [2].", [2]))
    assert status == "fail"
    assert "expected a cited source" in notes[0]


def test_fail_when_threshold_missing():
    status, notes = grade(CASE, _out("Düşük farkta oluşur [1].", [1]))
    assert status == "fail"
    assert "missing any of" in notes[0]


def test_fail_when_answer_cites_nothing():
    status, notes = grade(CASE, _out("Fark 8,6 K altında.", []))
    assert "answer cites no source" in notes


def test_refusing_a_covered_question_fails():
    assert grade(CASE, _out(rag.NO_ANSWER, []))[0] == "fail"


def test_refusal_case():
    assert grade(REFUSAL_CASE, _out(rag.NO_ANSWER, []))[0] == "pass"
    assert grade(REFUSAL_CASE, _out("Önce yağı boşaltın.", []))[0] == "fail"


def test_llm_failure_is_infra_error_not_fail():
    assert grade(CASE, _out("Cevap üretilemedi: 429", [], error=True))[0] == "infra_error"
