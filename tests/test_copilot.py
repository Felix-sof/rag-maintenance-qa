"""The copilot: rag.answer with tools, and the tools themselves (no LLM, no database)."""

import copilot_tools
import rag
from conftest import FakeGenerator, StubStore, make_hit

READINGS = {
    "product_type": "L", "air_temperature_k": 300.0, "process_temperature_k": 308.4,
    "rotational_speed_rpm": 1300, "torque_nm": 45.0, "tool_wear_min": 120,
}
HDF_DOC = make_hit(0.86, section="Arıza Modları > HDF", text="HDF: fark 8,6 K altında ve hız 1380 rpm altında.")


class TestTools:
    def test_assess_process_applies_the_rules_without_a_database(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = copilot_tools.call("assess_process", READINGS)
        assert result["triggered"] == ["HDF"]
        assert "similar_recorded_processes" not in result  # needs the dataset in Postgres

    def test_numbers_given_as_strings_are_accepted(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = copilot_tools.call("assess_process", {**READINGS, "torque_nm": "45"})
        assert result["inputs"]["torque_nm"] == 45.0

    def test_a_missing_reading_is_an_error_for_the_model_to_explain(self):
        args = {k: v for k, v in READINGS.items() if k != "torque_nm"}
        assert "torque_nm" in copilot_tools.call("assess_process", args)["error"]

    def test_bad_values_and_unknown_tools_never_raise(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert "error" in copilot_tools.call("assess_process", {**READINGS, "product_type": "Z"})
        assert "error" in copilot_tools.call("drop_tables", {})
        assert "error" in copilot_tools.call("query_dataset", {"query": "failure_rate_by_type"})  # no database

    def test_the_tool_schemas_offer_only_the_fixed_queries(self):
        query = next(t for t in copilot_tools.TOOLS if t["name"] == "query_dataset")
        assert set(query["parameters"]["properties"]["query"]["enum"]) == set(copilot_tools.process_data.QUERIES)


class TestAnswerWithTools:
    def test_a_tool_result_becomes_a_citable_source(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        gen = FakeGenerator(
            "HDF koşulları sağlanıyor: fark 8.4 K, sınır 8.6 K [T1][1]. Hız 1300 rpm [T1].",
            tool_calls=[("assess_process", READINGS)],
        )
        out = rag.answer("L tipi, 300 K, 308.4 K, 1300 rpm: risk?", retriever=StubStore([HDF_DOC]), generator=gen,
                         tools=copilot_tools)

        assert gen.tools_offered == ["assess_process", "query_dataset"]
        assert [c["name"] for c in out["tool_calls"]] == ["assess_process"]
        assert out["tool_calls"][0]["result"]["triggered"] == ["HDF"]
        assert (out["cited_tools"], out["cited"]) == ([1], [1])
        assert out["unsupported_numbers"] == []  # 8.4 and 1300 are in the tool result
        assert out["usage"]["llm_calls"] == 2  # the tool round and the answer
        assert rag.TOOL_RULES in gen.calls[0][0]

    def test_a_number_in_neither_the_passages_nor_the_tools_is_still_flagged(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        gen = FakeGenerator("Fark 7.9 K [T1].", tool_calls=[("assess_process", READINGS)])
        out = rag.answer("300 K, 308.4 K, 1300 rpm?", retriever=StubStore([HDF_DOC]), generator=gen, tools=copilot_tools)
        assert out["unsupported_numbers"] == ["7.9"]

    def test_without_a_tool_call_it_is_one_llm_call(self):
        gen = FakeGenerator("HDF düşük devirde oluşur [1].")
        out = rag.answer("HDF nedir?", retriever=StubStore([HDF_DOC]), generator=gen, tools=copilot_tools)
        assert out["tool_calls"] == [] and out["usage"]["llm_calls"] == 1

    def test_readings_get_past_the_off_topic_gate(self):
        """No passage matches bare readings well, but the tools can answer them."""
        weak = [make_hit(rag.MIN_SCORE - 0.1)]
        question = "L tipi: hava 300 K, proses 308.4 K, 1300 rpm, 45 Nm, 120 dk. Risk var mı?"
        with_tools = rag.answer(question, retriever=StubStore(weak), generator=FakeGenerator("..."), tools=copilot_tools)
        assert with_tools["outcome"] != "refused_gate"
        without = rag.answer(question, retriever=StubStore(weak), generator=FakeGenerator("..."))
        assert without["outcome"] == "refused_gate"

    def test_an_ordinary_off_topic_question_is_still_refused(self):
        gen = FakeGenerator()
        out = rag.answer("Pizza nasıl yapılır?", retriever=StubStore([make_hit(rag.MIN_SCORE - 0.1)]), generator=gen,
                         tools=copilot_tools)
        assert out["outcome"] == "refused_gate" and gen.calls == []


def test_readings_pattern():
    assert rag.gives_readings("Hava 27 °C, devir 1300 rpm")
    assert not rag.gives_readings("1380 rpm altında HDF olur mu?")  # one reading: a question about the rule
    assert not rag.gives_readings("Takım değişimi nasıl yapılır?")


def test_the_chat_ui_shows_the_tool_calls(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("DATABASE_URL", raising=False)
    assessment = copilot_tools.call("assess_process", READINGS)
    canned = {
        "question": "q", "search_query": "q", "answer": "HDF koşulları sağlanıyor [T1].", "sources": [], "cited": [],
        "tool_calls": [
            {"name": "assess_process", "args": READINGS, "result": assessment},
            {"name": "query_dataset", "args": {"query": "failure_rate_by_type"},
             "result": {"query": "failure_rate_by_type", "rows": [{"product_type": "L", "failure_rate": 0.0392}]}},
        ],
        "cited_tools": [1], "conflict": False, "unsupported_numbers": [], "error": False, "outcome": "answered",
    }
    monkeypatch.setattr(rag, "answer", lambda *args, **kwargs: canned)
    page = AppTest.from_file("../app.py", default_timeout=30).run()
    page.chat_input[0].set_value("L tipi, 300 K, 308.4 K, 1300 rpm, 45 Nm, 120 dk: risk?").run()
    assert not page.exception
    assert any("Tool calls (2)" in e.label for e in page.expander)
    assert len(page.dataframe) == 2  # the assessment table and the query's rows


def test_the_page_shows_refusals_and_errors_in_english(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def canned(answer, error=False):
        return {
            "question": "q", "search_query": "q", "answer": answer, "sources": [], "cited": [], "tool_calls": [],
            "cited_tools": [], "conflict": False, "unsupported_numbers": [], "error": error,
            "outcome": "error" if error else "refused_gate",
        }

    for result, shown in (
        (canned(rag.NO_ANSWER), "The knowledge base doesn't answer this question."),
        (canned(f"{rag.ERROR_PREFIX} Gemini quota exhausted", error=True), "Couldn't generate an answer: Gemini quota exhausted"),
    ):
        monkeypatch.setattr(rag, "answer", lambda *args, _result=result, **kwargs: _result)
        page = AppTest.from_file("../app.py", default_timeout=30).run()
        page.chat_input[0].set_value("Pizza?").run()
        assert not page.exception
        texts = [m.value for m in page.markdown] + [e.value for e in page.error]
        assert shown in texts, texts
