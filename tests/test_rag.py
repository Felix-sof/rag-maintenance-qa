import rag
from conftest import FakeGenerator, StubStore, make_hit
from generator import GenerationError

RELEVANT = [
    make_hit(0.89, section="Arızalar > HDF", text="HDF düşük devirde oluşur."),
    make_hit(0.85, source="safety.md", section="Kilitleme", doc_type="example", text="Şalteri kapatın."),
]


class TestExtractCitations:
    def test_single_and_grouped_citations(self):
        assert rag.extract_citations("A [1]. B [2, 3]. C [1][3].", 3) == [1, 2, 3]

    def test_out_of_range_numbers_are_ignored(self):
        assert rag.extract_citations("A [0]. B [4]. C [2].", 3) == [2]

    def test_no_citations(self):
        assert rag.extract_citations("Kaynaksız cevap.", 3) == []


def test_context_is_numbered_and_marks_example_documents():
    context = rag.format_context(RELEVANT)
    assert context.startswith("[1] failure_modes.md > Arızalar > HDF\nHDF düşük")
    assert "[2] safety.md > Kilitleme (örnek belge)" in context


class TestAnswer:
    def test_answers_from_context_and_reports_cited_sources(self):
        gen = FakeGenerator("HDF düşük devirde oluşur [1].")
        out = rag.answer("HDF nedir?", store=StubStore(RELEVANT), generator=gen)

        assert out["answer"] == "HDF düşük devirde oluşur [1]."
        assert out["cited"] == [1]
        assert [s["n"] for s in out["sources"]] == [1, 2]
        assert out["error"] is False

        system, prompt = gen.calls[0]
        assert system == rag.SYSTEM_PROMPT
        assert "[1] failure_modes.md > Arızalar > HDF" in prompt
        assert prompt.endswith("Question: HDF nedir?")

    def test_off_topic_question_is_refused_without_calling_the_llm(self):
        gen = FakeGenerator()
        out = rag.answer("Pizza tarifi?", store=StubStore([make_hit(rag.MIN_SCORE - 0.01)]), generator=gen)
        assert out["answer"] == rag.NO_ANSWER
        assert out["cited"] == []
        assert gen.calls == []

    def test_empty_index_is_refused(self):
        out = rag.answer("HDF?", store=StubStore([]), generator=FakeGenerator())
        assert out["answer"] == rag.NO_ANSWER

    def test_follow_up_is_condensed_before_retrieval(self):
        gen = FakeGenerator("Kesici takım değişiminin adımları nelerdir?", "Adımlar [1].")
        store = StubStore(RELEVANT)
        history = [{"question": "Takım ne zaman değiştirilir?", "answer": "Aşınınca [1]."}]

        out = rag.answer("Peki adımları neler?", history=history, store=store, generator=gen)

        assert store.queries == ["Kesici takım değişiminin adımları nelerdir?"]
        assert out["search_query"] == "Kesici takım değişiminin adımları nelerdir?"
        assert "Takım ne zaman değiştirilir?" in gen.calls[0][1]  # history went into the condense prompt
        assert gen.calls[1][1].endswith("Question: Kesici takım değişiminin adımları nelerdir?")

    def test_standalone_question_skips_condensing(self):
        gen = FakeGenerator("Cevap [1].")
        rag.answer("HDF nedir?", store=StubStore(RELEVANT), generator=gen)
        assert len(gen.calls) == 1

    def test_llm_failure_is_reported_not_raised(self):
        gen = FakeGenerator(error=GenerationError("Gemini API error (429): quota"))
        out = rag.answer("HDF nedir?", store=StubStore(RELEVANT), generator=gen)
        assert out["error"] is True
        assert "429" in out["answer"]

    def test_uses_process_wide_store_and_generator_by_default(self):
        # autouse fixture: real knowledge/ docs + FakeGenerator; hash-embedder scores are
        # far below MIN_SCORE, so this exercises the wiring, not answer quality
        out = rag.answer("HDF arızası")
        assert out["sources"] and out["sources"][0]["source"].endswith(".md")
