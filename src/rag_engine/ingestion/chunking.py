"""Character-window chunking with stable source offsets and metadata inheritance."""
from __future__ import annotations

import re

from rag_engine.config import ChunkingConfig
from rag_engine.schema import Chunk, Document


def chunk_document(document: Document, config: ChunkingConfig, strategy: str = "boundary") -> list[Chunk]:
    """Create overlapping chunks; boundary strategy prefers nearby whitespace or sentence breaks."""
    text = document.text
    if not text.strip():
        return []
    if strategy not in {"character", "boundary"}:
        raise ValueError("strategy must be 'character' or 'boundary'")
    output: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + config.max_chars)
        if strategy == "boundary" and end < len(text):
            lower_bound = start + int(config.max_chars * 0.65)
            boundary_candidates = [text.rfind("\n", lower_bound, end), text.rfind(". ", lower_bound, end), text.rfind(" ", lower_bound, end)]
            boundary = max(boundary_candidates)
            if boundary > lower_bound:
                end = boundary + (2 if text[boundary:boundary + 2] == ". " else 1)
        fragment = text[start:end].strip()
        if fragment:
            chunk_index = len(output)
            output.append(Chunk(
                chunk_id=f"{document.document_id}::chunk-{chunk_index:04d}",
                document_id=document.document_id,
                text=fragment,
                start_char=start,
                end_char=end,
                metadata={**document.metadata, "chunk_index": chunk_index},
            ))
        if end >= len(text):
            break
        next_start = max(start + 1, end - config.overlap_chars)
        # Move past any whitespace at the new boundary without skipping substantive text.
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return output
