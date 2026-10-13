"""Provider-neutral retrieval interface and metadata predicates."""
from __future__ import annotations

from typing import Protocol

from rag_engine.schema import Chunk, SearchHit


class Retriever(Protocol):
    def search(self, query: str, top_k: int, filters: dict[str, object] | None = None) -> list[SearchHit]: ...


def matches_metadata(chunk: Chunk, filters: dict[str, object] | None) -> bool:
    return all(chunk.metadata.get(key) == value for key, value in (filters or {}).items())
