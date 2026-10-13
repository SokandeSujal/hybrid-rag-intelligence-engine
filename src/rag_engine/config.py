"""Configuration dataclasses for ingestion, retrieval and generation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 600
    overlap_chars: int = 80

    def __post_init__(self) -> None:
        if self.max_chars <= 0 or self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("Require max_chars > overlap_chars >= 0")


@dataclass(frozen=True)
class RetrievalConfig:
    sparse_top_k: int = 10
    dense_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if min(self.sparse_top_k, self.dense_top_k, self.final_top_k, self.rrf_k) <= 0:
            raise ValueError("Retrieval depths and rrf_k must be positive")


@dataclass(frozen=True)
class ContextConfig:
    token_budget: int = 1200

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    minimum_evidence_score: float = 0.0
    abstain_when_empty: bool = True
