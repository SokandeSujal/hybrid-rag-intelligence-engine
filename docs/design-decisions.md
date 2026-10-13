# Design decisions and trade-offs

- Providers are protocols: data ingestion, dense retrieval, rewriting, reranking and generation can be swapped independently.
- An in-memory index keeps indexing mechanics inspectable. It is not persistent and does not enforce multi-user authorization.
- Feature hashing produces repeatable vectors without training weights; collisions and bag-of-token representations limit semantic quality.
- BM25 uses simple word tokenization and corpus-level document frequencies; stemming, language-specific analysis and phrase handling are extensions.
- RRF is the default score fusion because retriever score scales differ. It still depends on candidate depth and rank quality.
- The answer model sees only context selected by the pipeline. In a real system, system prompts, tool boundaries, injection resistance, source ACLs and output validation need explicit security review.
- Deterministic examples aid comprehension but cannot establish end-to-end answer correctness.
