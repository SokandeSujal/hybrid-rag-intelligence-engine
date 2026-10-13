"""Text, Markdown and optional PDF loaders with explicit dependency boundaries."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from rag_engine.schema import Document


class DocumentLoader(Protocol):
    def load(self, path: str | Path) -> Document: ...


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TextLoader:
    def load(self, path: str | Path) -> Document:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        text = normalize_text(source.read_text(encoding="utf-8"))
        return Document(source.stem, text, {"source_path": source.name, "format": "text"})


class MarkdownLoader(TextLoader):
    """Keep Markdown syntax in text while adding first-heading metadata."""
    def load(self, path: str | Path) -> Document:
        document = super().load(path)
        heading = next((line.lstrip("# ").strip() for line in document.text.splitlines() if line.startswith("#")), None)
        metadata = {**document.metadata, "format": "markdown"}
        if heading:
            metadata["title"] = heading
        return Document(document.document_id, document.text, metadata)


class PDFLoader:
    """Optional adapter; importing this module does not require pypdf to be installed."""
    def load(self, path: str | Path) -> Document:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF extraction requires the optional dependency: pip install .[pdf]") from exc
        reader = PdfReader(str(source))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = normalize_text("\n\n".join(pages))
        return Document(source.stem, text, {"source_path": source.name, "format": "pdf", "page_count": len(pages)})
