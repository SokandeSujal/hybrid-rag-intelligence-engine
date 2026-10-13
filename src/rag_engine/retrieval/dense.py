"""Dense retrieval protocol and deterministic feature-hashing reference retriever."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from rag_engine.retrieval.base import matches_metadata
from rag_engine.schema import Chunk, SearchHit


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.casefold())


def hashing_vector(text: str, dimension: int = 256) -> np.ndarray:
    """Turn terms into a signed hashed bag-of-words vector; not a learned embedding."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    vector = np.zeros(dimension, dtype=np.float32)
    for term in _terms(text):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


class DenseRetriever:
    """Protocol-shaped base class for providers returning cosine-like scores."""
    def search(self, query: str, top_k: int, filters: dict[str, object] | None = None) -> list[SearchHit]:
        raise NotImplementedError


class HashingDenseRetriever(DenseRetriever):
    def __init__(self, chunks: list[Chunk], dimension: int = 256) -> None:
        self.chunks = list(chunks)
        self.dimension = dimension
        self.matrix = np.stack([hashing_vector(chunk.text, dimension) for chunk in self.chunks]) if self.chunks else np.empty((0, dimension), dtype=np.float32)

    def search(self, query: str, top_k: int = 5, filters: dict[str, object] | None = None) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.chunks:
            return []
        query_vector = hashing_vector(query, self.dimension)
        scores = self.matrix @ query_vector
        candidates = [(index, float(score)) for index, score in enumerate(scores) if matches_metadata(self.chunks[index], filters)]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return [SearchHit(self.chunks[index], score, "hash_dense", rank + 1) for rank, (index, score) in enumerate(candidates[:top_k]) if score > 0]
