"""
Document loading and chunking.

Documents are split on their Markdown headings, so each chunk is one
coherent section (e.g. "HDF" or "Tool change procedure"), never the tail of
one topic glued to the head of another. The heading path is prepended to
each chunk's text, because a section body on its own ("Steps: 1. ...")
often doesn't say what it is about - which hurts both retrieval and the
LLM's reading of the context.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 1200
SUPPORTED_EXTENSIONS = (".md", ".txt")

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # file name, e.g. "failure_modes.md"
    section: str  # heading path, e.g. "Arıza Modları > HDF - Isı Dağıtım Arızası"
    title: str  # document title from front matter (falls back to the file name)
    doc_type: str  # "reference" (official documentation) or "example" (illustrative)


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


def chunk_document(text: str, source: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split a document into heading-scoped chunks of at most ~max_chars body text.

    Paragraphs are packed together up to max_chars but never across a
    section boundary. Plain-text files (no headings) become one section
    titled after the document.
    """
    meta, body = parse_front_matter(text)
    title = meta.get("title", source)
    doc_type = meta.get("type", "unknown")

    chunks = []
    for heading_path, section_body in _sections(body):
        section = " > ".join(heading_path) or title
        paragraphs = [p.strip() for p in section_body.split("\n\n") if p.strip()]

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

        for i, piece in enumerate(pieces):
            chunk_id = hashlib.sha1(f"{source}|{section}|{i}".encode("utf-8")).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=f"{section}\n\n{piece}",
                    source=source,
                    section=section,
                    title=title,
                    doc_type=doc_type,
                )
            )
    return chunks


def document_paths(docs_dir: Path) -> list[Path]:
    return sorted(p for p in Path(docs_dir).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_chunks(docs_dir: Path, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    chunks = []
    for path in document_paths(docs_dir):
        chunks.extend(chunk_document(path.read_text(encoding="utf-8"), path.name, max_chars))
    return chunks
