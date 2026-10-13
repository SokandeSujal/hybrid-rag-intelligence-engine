"""Stable data contracts shared across RAG stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    retriever: str
    rank: int = 0


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    query: str
    rewritten_queries: list[str]
    hits: list[SearchHit]
    evidence: list[Evidence]
    answer: GeneratedAnswer
