import pytest

from conftest import HashEmbedder
from retriever import Retriever, tokenize, turkish_lower
from vector_store import VectorStore


class FakeReranker:
    """Scores a candidate by how many times `favourite` appears in it."""

    name = "fake-reranker"

    def __init__(self, favourite):
        self.favourite = favourite
        self.seen = []

    def score(self, query, texts):
        self.seen.append(len(texts))
        return [float(t.count(self.favourite)) for t in texts]


@pytest.fixture
def store(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "arizalar.md").write_text(
        "# Arızalar\n\n## HDF\n\nIsı dağıtım arızası düşük devirde oluşur.\n\n"
        "## OSF\n\nAşırı zorlanma arızası OSF kodlu arızadır.\n\n"
        "## PWF\n\nGüç arızası tork ve devir çarpımı sınır dışına çıkınca oluşur.\n",
        encoding="utf-8",
    )
    return VectorStore(docs_dir=docs, index_dir=tmp_path / "idx", embedder=HashEmbedder())


class TestTokenize:
    def test_turkish_dotted_and_dotless_i(self):
        assert turkish_lower("ISI İŞLEM") == "ısı işlem"

    def test_suffixes_collapse_to_the_same_stem(self):
        assert tokenize("arıza arızası arızaları")[0] == tokenize("arızanın")[0] == "arıza"

    def test_stopwords_are_dropped(self):
        assert tokenize("HDF nedir ve nasıl oluşur") == ["hdf", "oluşu"]


class TestModes:
    def test_unknown_mode_is_rejected(self, store):
        with pytest.raises(ValueError):
            Retriever(store=store, mode="magic")

    def test_bm25_mode_finds_exact_code(self, store):
        hits = Retriever(store=store, mode="bm25").search("OSF", top_k=1)
        assert hits[0]["section"] == "Arızalar > OSF"
        assert hits[0]["ranks"]["bm25"] == 1

    def test_hybrid_returns_hits_from_both_rankings_with_ranks(self, store):
        hits = Retriever(store=store, mode="hybrid").search("OSF arızası", top_k=3)
        assert {h["section"] for h in hits} == {"Arızalar > HDF", "Arızalar > OSF", "Arızalar > PWF"}
        assert hits[0]["section"] == "Arızalar > OSF"  # top of both rankings -> top after fusion
        assert all(h["ranks"]["vector"] for h in hits)

    def test_reranker_has_the_final_say(self, store):
        reranker = FakeReranker(favourite="tork")
        hits = Retriever(store=store, mode="hybrid_rerank", reranker=reranker).search("OSF", top_k=3)
        assert hits[0]["section"] == "Arızalar > PWF"
        assert hits[0]["ranks"]["rerank"] == 1.0
        assert reranker.seen == [3]  # only the fused shortlist is re-scored

    def test_score_is_cosine_in_every_mode(self, store):
        by_mode = {
            mode: {h["id"]: h["score"] for h in Retriever(store=store, mode=mode).search("güç arızası", top_k=3)}
            for mode in ("vector", "hybrid")
        }
        assert by_mode["vector"] == by_mode["hybrid"]


def test_keyword_index_follows_a_rebuild(store):
    retriever = Retriever(store=store, mode="bm25")
    assert retriever.search("RNF", top_k=1) == []
    (store.docs_dir / "rnf.md").write_text("# RNF\n\nRNF rastgele arızadır.\n", encoding="utf-8")
    store.build()
    assert retriever.search("RNF", top_k=1)[0]["source"] == "rnf.md"


def test_empty_corpus(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    store = VectorStore(docs_dir=empty, index_dir=tmp_path / "idx", embedder=HashEmbedder())
    assert Retriever(store=store, mode="hybrid").search("herhangi") == []
