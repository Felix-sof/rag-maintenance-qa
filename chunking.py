"""
Document loading and chunking.

Documents are split on their Markdown headings, so each chunk is one
coherent section (e.g. "HDF" or "Tool change procedure"), never the tail of
one topic glued to the head of another. The heading path is prepended to
each chunk's text, because a section body on its own ("Steps: 1. ...")
often doesn't say what it is about - which hurts both retrieval and the
LLM's reading of the context.

PDFs have no headings to split on, so each page becomes a section
("manual.pdf > s. 3"), which also makes citations point at a page.

Metadata (title, type, priority, updated) comes from, in order of
precedence: a sidecar "<file>.meta.json" (written by the upload UI, and the
only option for PDFs), the document's front matter, then defaults.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 1200

# When documents conflict, the higher priority wins (then the newer `updated`
# date). Metadata can set `priority` explicitly; otherwise it follows the type.
DEFAULT_PRIORITY = {"reference": 2, "example": 1}
SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf")
META_SUFFIX = ".meta.json"

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # file name, e.g. "failure_modes.md"
    section: str  # heading path, e.g. "Arıza Modları > HDF - Isı Dağıtım Arızası"
    title: str  # document title from front matter (falls back to the file name)
    doc_type: str  # "reference" (official documentation) or "example" (illustrative)
    priority: int = 1  # higher wins when documents conflict
    updated: str = ""  # ISO date from front matter, "" if unknown


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a minimal `---\\nkey: value\\n---` header off a document."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[3:end].strip().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, text[end + len("\n---") :].lstrip("\n")


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split one over-long paragraph on sentence boundaries (hard cut as a last resort)."""
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _sections(body: str) -> list[tuple[list[str], str]]:
    """Group document lines into (heading path, section body) pairs."""
    sections = []
    heading_stack: list[tuple[int, str]] = []
    lines: list[str] = []

    def flush():
        if any(line.strip() for line in lines):
            sections.append(([h for _, h in heading_stack], "\n".join(lines)))
        lines.clear()

    for line in body.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading_stack = [(lvl, h) for lvl, h in heading_stack if lvl < level]
            heading_stack.append((level, match.group(2)))
        else:
            lines.append(line)
    flush()
    return sections


def _pack(paragraphs: list[str], max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most ~max_chars, splitting over-long ones."""
    pieces, current = [], ""
    for paragraph in paragraphs:
        parts = _split_long(paragraph, max_chars) if len(paragraph) > max_chars else [paragraph]
        for part in parts:
            if current and len(current) + 2 + len(part) > max_chars:
                pieces.append(current)
                current = part
            else:
                current = f"{current}\n\n{part}" if current else part
    if current:
        pieces.append(current)
    return pieces


def _resolve_meta(meta: dict, source: str) -> dict:
    doc_type = meta.get("type", "unknown")
    try:
        priority = int(meta["priority"])
    except (KeyError, TypeError, ValueError):
        priority = DEFAULT_PRIORITY.get(doc_type, 1)
    return {
        "title": meta.get("title") or source,
        "doc_type": doc_type,
        "priority": priority,
        "updated": str(meta.get("updated", "")),
    }


def _make_chunks(sections: list[tuple[str, list[str]]], source: str, meta: dict, max_chars: int) -> list[Chunk]:
    chunks = []
    for section, paragraphs in sections:
        for i, piece in enumerate(_pack(paragraphs, max_chars)):
            chunk_id = hashlib.sha1(f"{source}|{section}|{i}".encode("utf-8")).hexdigest()[:16]
            chunks.append(Chunk(id=chunk_id, text=f"{section}\n\n{piece}", source=source, section=section, **meta))
    return chunks


def chunk_document(text: str, source: str, max_chars: int = MAX_CHUNK_CHARS, overrides: dict = None) -> list[Chunk]:
    """Split a Markdown/text document into heading-scoped chunks of at most ~max_chars body text.

    Paragraphs are packed together up to max_chars but never across a
    section boundary. Plain-text files (no headings) become one section
    titled after the document.
    """
    front_matter, body = parse_front_matter(text)
    meta = _resolve_meta({**front_matter, **(overrides or {})}, source)
    sections = [
        (" > ".join(heading_path) or meta["title"], [p.strip() for p in section_body.split("\n\n") if p.strip()])
        for heading_path, section_body in _sections(body)
    ]
    return _make_chunks(sections, source, meta, max_chars)


def chunk_pdf(path: Path, source: str, max_chars: int = MAX_CHUNK_CHARS, overrides: dict = None) -> list[Chunk]:
    """Split a PDF into page-scoped chunks ("<title> > s. <page>").

    Only the text layer is read: a scanned PDF without one yields no chunks
    (it would need OCR first).
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pdf_title = (reader.metadata.title if reader.metadata else None) or Path(source).stem
    meta = _resolve_meta({"title": pdf_title, **(overrides or {})}, source)

    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\r", "")
        # PDF text often has hard line breaks inside paragraphs; keep blank-line breaks only
        paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
        paragraphs = [p for p in paragraphs if p]
        if paragraphs:
            sections.append((f"{meta['title']} > s. {page_number}", paragraphs))
    return _make_chunks(sections, source, meta, max_chars)


def read_sidecar(path: Path) -> dict:
    sidecar = path.with_name(path.name + META_SUFFIX)
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def document_paths(docs_dir: Path) -> list[Path]:
    """Every supported document under docs_dir, including subfolders (e.g. uploads/)."""
    return sorted(p for p in Path(docs_dir).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def source_name(path: Path, docs_dir: Path) -> str:
    return path.relative_to(docs_dir).as_posix()


def load_document(path: Path, docs_dir: Path, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    source = source_name(path, docs_dir)
    overrides = read_sidecar(path)
    if path.suffix.lower() == ".pdf":
        return chunk_pdf(path, source, max_chars, overrides)
    return chunk_document(path.read_text(encoding="utf-8"), source, max_chars, overrides)


def load_chunks(docs_dir: Path, max_chars: int = MAX_CHUNK_CHARS, errors: dict = None) -> list[Chunk]:
    """Chunk every document. An unreadable one (corrupt PDF, bad encoding) is skipped,
    not fatal - its source and error go into `errors` if a dict is passed."""
    chunks = []
    for path in document_paths(docs_dir):
        try:
            chunks.extend(load_document(path, Path(docs_dir), max_chars))
        except Exception as exc:  # noqa: BLE001 - one bad upload must not take down the whole index
            if errors is not None:
                errors[source_name(path, Path(docs_dir))] = f"{type(exc).__name__}: {exc}"
    return chunks
