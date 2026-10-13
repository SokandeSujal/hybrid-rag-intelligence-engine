"""Evidence selection with an approximate, replaceable token counter."""
from __future__ import annotations

from collections.abc import Callable

from rag_engine.schema import Evidence, SearchHit


def approximate_tokens(text: str) -> int:
    """Rough word-based estimate, not a guarantee for any particular tokenizer."""
    return max(1, len(text.split())) if text.strip() else 0


def select_context(hits: list[SearchHit], token_budget: int,
                   token_counter: Callable[[str], int] = approximate_tokens) -> list[Evidence]:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    selected: list[Evidence] = []
    used = 0
    seen_documents: set[tuple[str, str]] = set()
    for hit in hits:
        key = (hit.chunk.document_id, hit.chunk.chunk_id)
        if key in seen_documents:
            continue
        cost = token_counter(hit.chunk.text)
        if cost <= 0:
            continue
        if used + cost > token_budget:
            continue
        selected.append(Evidence(hit.chunk.chunk_id, hit.chunk.document_id, hit.chunk.text, hit.score, hit.chunk.metadata))
        seen_documents.add(key)
        used += cost
    return selected
