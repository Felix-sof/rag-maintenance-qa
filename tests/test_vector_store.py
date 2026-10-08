import pytest

import vector_store
from conftest import HashEmbedder
from vector_store import VectorStore


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "arizalar.md").write_text(
        "# Arızalar\n\n## HDF\n\nIsı dağıtım arızası düşük devirde oluşur.\n\n"
        "## PWF\n\nGüç arızası tork ve devir çarpımı sınır dışına çıkınca oluşur.\n",
        encoding="utf-8",
    )
    (d / "guvenlik.md").write_text(
        "---\ntype: example\n---\n\n# Kilitleme\n\nAna şalteri kapatın ve kilidinizi takın.\n",
        encoding="utf-8",
    )
    return d


def make_store(docs, tmp_path):
    return VectorStore(docs_dir=docs, index_dir=tmp_path / "idx", embedder=HashEmbedder())


class TestBuild:
    def test_indexes_every_chunk(self, docs, tmp_path):
        store = make_store(docs, tmp_path)
        assert store.build() == {"rebuilt": True, "chunks": 3}
        assert store.stats()["documents"] == 2

    def test_unchanged_documents_reuse_the_index(self, docs, tmp_path):
        store = make_store(docs, tmp_path)
        store.build()
        assert store.build()["rebuilt"] is False

    def test_index_persists_across_instances(self, docs, tmp_path):
        make_store(docs, tmp_path).build()
        assert make_store(docs, tmp_path).build()["rebuilt"] is False

    def test_editing_a_document_triggers_rebuild(self, docs, tmp_path):
        store = make_store(docs, tmp_path)
        store.build()
        (docs / "guvenlik.md").write_text("# Yeni\n\nYeni içerik.\n", encoding="utf-8")
        assert store.build()["rebuilt"] is True

    def test_adding_a_document_triggers_rebuild(self, docs, tmp_path):
        store = make_store(docs, tmp_path)
        store.build()
        (docs / "yeni.txt").write_text("yeni belge", encoding="utf-8")
        assert store.build() == {"rebuilt": True, "chunks": 4}

    def test_force_rebuilds_even_when_unchanged(self, docs, tmp_path):
        store = make_store(docs, tmp_path)
        store.build()
        assert store.build(force=True)["rebuilt"] is True


class TestSearch:
    def test_best_match_ranks_first(self, docs, tmp_path):
        hits = make_store(docs, tmp_path).search("tork ve devir çarpımı güç arızası", top_k=2)
        assert hits[0]["section"] == "Arızalar > PWF"
        assert hits[0]["score"] >= hits[1]["score"]

    def test_hits_carry_citation_metadata(self, docs, tmp_path):
        hit = make_store(docs, tmp_path).search("şalteri kapatın kilidinizi", top_k=1)[0]
        assert {k: hit[k] for k in ("source", "section", "doc_type")} == {
            "source": "guvenlik.md",
            "section": "Kilitleme",
            "doc_type": "example",
        }

    def test_top_k_is_clamped(self, docs, tmp_path):
        store = make_store(docs, tmp_path)
        assert len(store.search("arıza", top_k=50)) == 3
        assert len(store.search("arıza", top_k=0)) == 1

    def test_empty_docs_dir_returns_no_hits(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert make_store(empty, tmp_path).search("herhangi bir şey") == []


def test_real_knowledge_base_is_indexable():
    # the autouse fixture points get_store() at knowledge/ with the hash embedder
    stats = vector_store.get_store().stats()
    assert stats["documents"] == 5
    assert stats["chunks"] > 20
