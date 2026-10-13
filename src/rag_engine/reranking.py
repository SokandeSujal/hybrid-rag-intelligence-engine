"""Reranker interface and transparent lexical-overlap baseline."""
from __future__ import annotations

import re
from typing import Protocol

from rag_engine.schema import SearchHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


class LexicalOverlapReranker:
    def rerank(self, query: str, hits: list[SearchHit], top_k: int = 5) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = set(re.findall(r"[a-z0-9_]+", query.casefold()))
        rescored = []
        for hit in hits:
            document_terms = set(re.findall(r"[a-z0-9_]+", hit.chunk.text.casefold()))
            overlap = len(terms & document_terms) / max(1, len(terms))
            rescored.append(SearchHit(hit.chunk, overlap + 1e-6 * hit.score, "lexical_rerank", hit.rank))
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return [SearchHit(hit.chunk, hit.score, hit.retriever, rank + 1) for rank, hit in enumerate(rescored[:top_k])]
