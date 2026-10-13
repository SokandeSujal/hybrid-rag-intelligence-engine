# Implementation guide

Start with the schema so all modules share source IDs and metadata. Next compare the BM25 term-frequency score with the hash-vector cosine similarity. `HybridRetriever` fuses rank positions using `1 / (rrf_k + rank)`, where ranks are one-based. This intentionally avoids raw-score scale assumptions.

Context selection estimates token count with whitespace-separated words. That makes the example deterministic, but it is not safe for exact context-window accounting. For a deployed generator, use the exact model tokenizer and reserve space for system instructions and generated output. Always track source identifiers through every transformation.

The built-in `MockGroundedGenerator` selects a short evidence excerpt and returns its chunk ID. It does not establish that the evidence answers every question correctly. Use it to study the interface and abstention path, then replace it with a reviewed provider adapter.

## Evaluation notes

Retrieval tests need a query and set of relevant chunk IDs. Recall@k checks how much of the relevant set is retrieved; MRR emphasizes the first relevant hit; nDCG handles graded relevance if labels are available. Faithfulness and citation quality require human review for reliable claims. The included lexical proxies can highlight obvious failures, but they are not substitutes for adjudicated evaluation.
