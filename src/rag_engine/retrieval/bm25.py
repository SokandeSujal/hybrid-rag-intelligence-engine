"""Small, transparent in-memory BM25 retriever."""
from __future__ import annotations

import math
import re
from collections import Counter

from rag_engine.retrieval.base import matches_metadata
from rag_engine.schema import Chunk, SearchHit


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.casefold())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require k1 > 0 and b in [0, 1]")
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        self.idf = {
            term: math.log(1.0 + (len(self.chunks) - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5, filters: dict[str, object] | None = None) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = tokenize(query)
        if not query_terms:
            return []
        hits: list[SearchHit] = []
        for index, chunk in enumerate(self.chunks):
            if not matches_metadata(chunk, filters):
                continue
            length = self.lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = self.term_counts[index].get(term, 0)
                if not frequency:
                    continue
                norm = frequency + self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1e-8))
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / norm
            if score > 0:
                hits.append(SearchHit(chunk, score, "bm25"))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [SearchHit(hit.chunk, hit.score, hit.retriever, rank + 1) for rank, hit in enumerate(hits[:top_k])]
