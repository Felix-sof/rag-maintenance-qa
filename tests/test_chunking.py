from chunking import chunk_document, load_chunks, parse_front_matter

DOC = """---
title: Test Belgesi
type: reference
---

# Arızalar

Giriş paragrafı.

## HDF

Isı dağıtım arızası düşük devir ve düşük sıcaklık farkında oluşur.

## PWF

Güç arızası tork ve devir çarpımı sınır dışına çıkınca oluşur.

### Alt başlık

Alt başlık içeriği.
"""


class TestFrontMatter:
    def test_parses_metadata_and_strips_header(self):
        meta, body = parse_front_matter(DOC)
        assert meta == {"title": "Test Belgesi", "type": "reference"}
        assert body.startswith("# Arızalar")

    def test_document_without_front_matter_is_unchanged(self):
        assert parse_front_matter("# Başlık\n\nmetin") == ({}, "# Başlık\n\nmetin")


class TestChunkDocument:
    def test_one_chunk_per_section_with_heading_path(self):
        assert [c.section for c in chunk_document(DOC, "a.md")] == [
            "Arızalar",
            "Arızalar > HDF",
            "Arızalar > PWF",
            "Arızalar > PWF > Alt başlık",
        ]

    def test_heading_path_is_prepended_to_text(self):
        hdf = chunk_document(DOC, "a.md")[1]
        assert hdf.text.startswith("Arızalar > HDF\n\n")
        assert "Isı dağıtım" in hdf.text

    def test_sibling_heading_resets_deeper_levels(self):
        doc = "# A\n\n## B\n\n### C\n\nx\n\n## D\n\ny\n"
        assert [c.section for c in chunk_document(doc, "d.md")] == ["A > B > C", "A > D"]

    def test_metadata_is_carried_onto_chunks(self):
        chunk = chunk_document(DOC, "a.md")[0]
        assert (chunk.source, chunk.title, chunk.doc_type) == ("a.md", "Test Belgesi", "reference")

    def test_plain_text_without_headings_is_one_section_named_after_the_file(self):
        chunks = chunk_document("Sadece düz metin.\n\nİkinci paragraf.", "notlar.txt")
        assert len(chunks) == 1
        assert chunks[0].section == "notlar.txt"
        assert chunks[0].doc_type == "unknown"

    def test_long_section_is_split_under_max_chars(self):
        sentences = " ".join(f"Bu {i}. cümledir." for i in range(200))
        chunks = chunk_document(f"# Uzun\n\n{sentences}\n", "u.md", max_chars=300)
        assert len(chunks) > 1
        assert all(len(c.text) - len("Uzun\n\n") <= 300 for c in chunks)

    def test_chunk_ids_are_unique_and_stable(self):
        first = [c.id for c in chunk_document(DOC, "a.md")]
        assert first == [c.id for c in chunk_document(DOC, "a.md")]
        assert len(set(first)) == len(first)


def test_load_chunks_reads_md_and_txt_only(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nmetin", encoding="utf-8")
    (tmp_path / "b.txt").write_text("düz metin", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4")
    assert sorted(c.source for c in load_chunks(tmp_path)) == ["a.md", "b.txt"]
