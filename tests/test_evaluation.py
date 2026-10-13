"""Illustrative metric tests; not executed during archive creation."""
from rag_engine.evaluation.metrics import retrieval_metrics
from rag_engine.schema import Chunk, SearchHit


def test_retrieval_metric_values() -> None:
    chunk = Chunk("relevant", "doc", "evidence", 0, 8)
    metrics = retrieval_metrics([SearchHit(chunk, 1.0, "test", 1)], {"relevant"}, k=1)
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg@1"] == 1.0
