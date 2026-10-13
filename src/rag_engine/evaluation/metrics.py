"""Transparent ranking and heuristic answer-grounding diagnostics."""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

from rag_engine.schema import Evidence, GeneratedAnswer, SearchHit


def retrieval_metrics(ranked_hits: Sequence[SearchHit], relevant_ids: set[str], k: int = 5,
                      graded_relevance: dict[str, float] | None = None) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return {f"recall@{k}": 0.0, "mrr": 0.0, f"ndcg@{k}": 0.0}
    top = list(ranked_hits[:k])
    found = {hit.chunk.chunk_id for hit in top}
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(ranked_hits, start=1) if hit.chunk.chunk_id in relevant_ids), 0.0)
    gains = [float((graded_relevance or {}).get(hit.chunk.chunk_id, 1.0 if hit.chunk.chunk_id in relevant_ids else 0.0)) for hit in top]
    dcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_gains = sorted((float((graded_relevance or {}).get(key, 1.0)) for key in relevant_ids), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return {f"recall@{k}": len(found & relevant_ids) / len(relevant_ids), "mrr": reciprocal_rank, f"ndcg@{k}": dcg / idcg if idcg else 0.0}


def citation_quality(answer: GeneratedAnswer, evidence: Sequence[Evidence]) -> dict[str, float]:
    available = {item.chunk_id for item in evidence}
    citations = answer.citations
    valid = sum(citation in available for citation in citations)
    precision = valid / len(citations) if citations else (1.0 if answer.abstained else 0.0)
    return {"citation_precision": precision, "has_citation": float(bool(citations)), "abstained": float(answer.abstained)}


def lexical_faithfulness_proxy(answer_text: str, evidence: Sequence[Evidence]) -> float:
    """A rough content-word overlap proxy; not a semantic or human faithfulness judgment."""
    words = lambda text: {word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) > 2}
    answer_words = words(answer_text)
    evidence_words = set().union(*(words(item.text) for item in evidence)) if evidence else set()
    return len(answer_words & evidence_words) / max(1, len(answer_words))
