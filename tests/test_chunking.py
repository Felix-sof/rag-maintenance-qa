from chunking import chunk_document, chunk_pdf, load_chunks, parse_front_matter
from conftest import make_pdf

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


class TestPriority:
    def test_explicit_priority_and_updated_date(self):
        doc = "---\ntype: example\npriority: 5\nupdated: 2026-09-01\n---\n\n# A\n\nmetin\n"
        chunk = chunk_document(doc, "a.md")[0]
        assert (chunk.priority, chunk.updated) == (5, "2026-09-01")

    def test_priority_defaults_follow_document_type(self):
        assert chunk_document("---\ntype: reference\n---\n\n# A\n\nx\n", "a.md")[0].priority == 2
        assert chunk_document("---\ntype: example\n---\n\n# A\n\nx\n", "a.md")[0].priority == 1
        assert chunk_document("# A\n\nx\n", "a.md")[0].priority == 1

    def test_invalid_priority_falls_back_to_default(self):
        chunk = chunk_document("---\ntype: reference\npriority: yüksek\n---\n\n# A\n\nx\n", "a.md")[0]
        assert chunk.priority == 2


class TestPdf:
    def test_one_section_per_page_titled_from_metadata(self, tmp_path):
        path = tmp_path / "pompa.pdf"
        path.write_bytes(make_pdf(["Pump pressure check\nOpen valve A", "Replace the filter"], title="Pompa Kilavuzu"))
        chunks = chunk_pdf(path, "pompa.pdf")
        assert [c.section for c in chunks] == ["Pompa Kilavuzu > s. 1", "Pompa Kilavuzu > s. 2"]
        assert "Pump pressure check" in chunks[0].text
        assert "Replace the filter" in chunks[1].text

    def test_title_falls_back_to_file_name_and_blank_pages_are_skipped(self, tmp_path):
        path = tmp_path / "kilavuz.pdf"
        path.write_bytes(make_pdf(["", "Only page two has text"]))
        assert [c.section for c in chunk_pdf(path, "kilavuz.pdf")] == ["kilavuz > s. 2"]

    def test_sidecar_metadata_applies_to_pdf(self, tmp_path):
        (tmp_path / "k.pdf").write_bytes(make_pdf(["text"]))
        (tmp_path / "k.pdf.meta.json").write_text(
            '{"type": "reference", "priority": 4, "updated": "2026-09-01"}', encoding="utf-8"
        )
        chunk = load_chunks(tmp_path)[0]
        assert (chunk.doc_type, chunk.priority, chunk.updated) == ("reference", 4, "2026-09-01")


class TestLoadChunks:
    def test_reads_supported_files_including_subfolders(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n\nmetin", encoding="utf-8")
        (tmp_path / "uploads").mkdir()
        (tmp_path / "uploads" / "b.txt").write_text("düz metin", encoding="utf-8")
        (tmp_path / "c.docx").write_bytes(b"not supported")
        assert sorted(c.source for c in load_chunks(tmp_path)) == ["a.md", "uploads/b.txt"]

    def test_unreadable_document_is_skipped_and_reported(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n\nmetin", encoding="utf-8")
        (tmp_path / "bozuk.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
        errors = {}
        assert [c.source for c in load_chunks(tmp_path, errors=errors)] == ["a.md"]
        assert list(errors) == ["bozuk.pdf"]

    def test_sidecar_overrides_front_matter(self, tmp_path):
        (tmp_path / "a.md").write_text("---\ntype: example\n---\n\n# A\n\nx\n", encoding="utf-8")
        (tmp_path / "a.md.meta.json").write_text('{"type": "reference"}', encoding="utf-8")
        chunk = load_chunks(tmp_path)[0]
        assert (chunk.doc_type, chunk.priority) == ("reference", 2)
