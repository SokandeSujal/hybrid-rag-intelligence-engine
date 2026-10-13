"""Reciprocal-rank fusion across independent retrieval sources."""
from __future__ import annotations

from collections import defaultdict

from rag_engine.retrieval.base import Retriever
from rag_engine.schema import SearchHit


class HybridRetriever:
    def __init__(self, sparse: Retriever, dense: Retriever, rrf_k: int = 60) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.sparse, self.dense, self.rrf_k = sparse, dense, rrf_k

    def search(self, query: str, top_k: int = 5, filters: dict[str, object] | None = None,
               sparse_top_k: int | None = None, dense_top_k: int | None = None) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        sparse_hits = self.sparse.search(query, sparse_top_k or top_k, filters)
        dense_hits = self.dense.search(query, dense_top_k or top_k, filters)
        by_id = {}
        scores: defaultdict[str, float] = defaultdict(float)
        origins: defaultdict[str, set[str]] = defaultdict(set)
        for hits in (sparse_hits, dense_hits):
            for rank, hit in enumerate(hits, start=1):
                key = hit.chunk.chunk_id
                by_id[key] = hit.chunk
                scores[key] += 1.0 / (self.rrf_k + rank)
                origins[key].add(hit.retriever)
        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [SearchHit(by_id[key], scores[key], "+".join(sorted(origins[key])), rank + 1) for rank, key in enumerate(ordered)]
