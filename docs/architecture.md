# Architecture and contracts

## Core schema

`Document` represents an ingested source. `Chunk` carries a stable chunk ID, parent document ID, text span, and inherited metadata. Retrieval components return `SearchHit` records with the chunk, score, and retriever origin. `Evidence` is the source-citable unit passed into context selection and generation. The generated answer carries citation IDs and an explicit `abstained` flag.

## Query-time path

1. The query rewriter returns one or more query strings; the default returns the original query.
2. Sparse and dense retrievers independently return ranked hits, with metadata equality filters applied before results are emitted.
3. Hybrid retrieval merges the rankings using reciprocal-rank fusion. Duplicate chunks are merged rather than counted twice.
4. The reranker assigns a new relevance estimate to existing candidates. It cannot discover evidence absent from the candidate pool.
5. Context selection packs evidence up to a configured approximate token budget and preserves citation IDs.
6. The generator receives the question and the exact selected evidence list. It may return an answer with citations or an abstention.
7. Evaluation reports retrieval ranking separately from answer grounding and source attribution.

## Error boundaries

Loaders raise clear errors for missing files and unsupported PDF extraction dependencies. Retrieval on an empty corpus returns no hits. The pipeline treats no evidence as a first-class state. Provider adapters should add timeouts, retries with bounded policies, privacy review, structured logs and secret management outside the core modules.
