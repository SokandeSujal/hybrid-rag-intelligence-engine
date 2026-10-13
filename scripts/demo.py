"""Offline synthetic RAG demonstration (not executed during packaging)."""
from __future__ import annotations

import json
from pathlib import Path

from rag_engine.config import ChunkingConfig, GenerationConfig
from rag_engine.generation import MockGroundedGenerator
from rag_engine.ingestion.chunking import chunk_document
from rag_engine.pipeline import RAGPipeline
from rag_engine.retrieval.bm25 import BM25Retriever
from rag_engine.retrieval.dense import HashingDenseRetriever
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.reranking import LexicalOverlapReranker
from rag_engine.schema import Document


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "examples" / "synthetic_documents.json"
    config = json.loads((project_root / "configs" / "demo.json").read_text(encoding="utf-8"))
    records = json.loads(source.read_text(encoding="utf-8"))
    chunks = []
    chunk_config = ChunkingConfig(**config["chunking"])
    for item in records:
        document = Document(item["id"], item["text"], item["metadata"])
        chunks.extend(chunk_document(document, chunk_config))
    retrieval = config["retrieval"]
    hybrid = HybridRetriever(BM25Retriever(chunks), HashingDenseRetriever(chunks), rrf_k=int(retrieval["rrf_k"]))
    pipeline = RAGPipeline(
        hybrid, MockGroundedGenerator(GenerationConfig(**config["generation"])),
        reranker=LexicalOverlapReranker(), context_budget=int(config["context"]["token_budget"]),
        candidate_k=max(int(retrieval["sparse_top_k"]), int(retrieval["dense_top_k"])),
        final_k=int(retrieval["final_top_k"]),
    )
    result = pipeline.run("How should retrieval ranking and answer quality be evaluated?")
    print(json.dumps({"answer": result.answer.answer, "citations": result.answer.citations, "abstained": result.answer.abstained,
                      "retrieved_chunk_ids": [hit.chunk.chunk_id for hit in result.hits]}, indent=2))


if __name__ == "__main__":
    main()
