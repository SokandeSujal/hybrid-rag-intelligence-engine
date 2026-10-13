"""Illustrative retrieval tests; not executed during archive creation."""
from rag_engine.retrieval.bm25 import BM25Retriever
from rag_engine.retrieval.dense import HashingDenseRetriever
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.schema import Chunk


def sample_chunks() -> list[Chunk]:
    return [Chunk("a", "doc-a", "validation data and held out test set", 0, 38, {"topic": "ml"}),
            Chunk("b", "doc-b", "hybrid retrieval fuses sparse and dense rankings", 0, 48, {"topic": "rag"})]


def test_bm25_metadata_filter() -> None:
    results = BM25Retriever(sample_chunks()).search("validation data", filters={"topic": "ml"})
    assert results and results[0].chunk.chunk_id == "a"


def test_hybrid_deduplicates_chunks() -> None:
    chunks = sample_chunks()
    retriever = HybridRetriever(BM25Retriever(chunks), HashingDenseRetriever(chunks))
    results = retriever.search("hybrid retrieval rankings", top_k=2)
    assert len({hit.chunk.chunk_id for hit in results}) == len(results)
