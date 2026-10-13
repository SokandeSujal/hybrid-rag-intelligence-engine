"""End-to-end RAG orchestration with explicit evidence and answer contracts."""
from __future__ import annotations

import logging

from rag_engine.context import select_context
from rag_engine.observability import emit_event
from rag_engine.generation import AnswerGenerator
from rag_engine.query import IdentityQueryRewriter, QueryRewriter
from rag_engine.reranking import LexicalOverlapReranker, Reranker
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.schema import PipelineResult, SearchHit


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: AnswerGenerator,
                 query_rewriter: QueryRewriter | None = None, reranker: Reranker | None = None,
                 context_budget: int = 1200, candidate_k: int = 10, final_k: int = 5) -> None:
        if min(context_budget, candidate_k, final_k) <= 0:
            raise ValueError("Context budget and retrieval depths must be positive")
        self.retriever = retriever
        self.generator = generator
        self.query_rewriter = query_rewriter or IdentityQueryRewriter()
        self.reranker = reranker or LexicalOverlapReranker()
        self.context_budget, self.candidate_k, self.final_k = context_budget, candidate_k, final_k
        self.logger = logging.getLogger("rag_engine.pipeline")

    def run(self, query: str, filters: dict[str, object] | None = None) -> PipelineResult:
        rewritten = self.query_rewriter.rewrite(query)
        emit_event(self.logger, "query_rewritten", query_length=len(query), variant_count=len(rewritten))
        all_hits: dict[str, SearchHit] = {}
        reciprocal_scores: dict[str, float] = {}
        for rewritten_query in rewritten:
            for rank, hit in enumerate(self.retriever.search(rewritten_query, self.candidate_k, filters), start=1):
                key = hit.chunk.chunk_id
                all_hits[key] = hit
                reciprocal_scores[key] = reciprocal_scores.get(key, 0.0) + 1.0 / (60 + rank)
        merged = [SearchHit(hit.chunk, reciprocal_scores[key], "multi_query_fusion", rank + 1)
                  for rank, key in enumerate(sorted(all_hits, key=reciprocal_scores.get, reverse=True))
                  for hit in [all_hits[key]]]
        reranked = self.reranker.rerank(query, merged, self.final_k)
        evidence = select_context(reranked, self.context_budget)
        answer = self.generator.generate(query, evidence)
        emit_event(self.logger, "rag_run_complete", candidate_count=len(merged), selected_evidence_count=len(evidence),
                   citation_count=len(answer.citations), abstained=answer.abstained)
        return PipelineResult(query, rewritten, reranked, evidence, answer)
