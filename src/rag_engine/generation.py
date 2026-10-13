"""Provider-independent answer generation with explicit evidence/abstention inputs."""
from __future__ import annotations

import re
from typing import Protocol

from rag_engine.config import GenerationConfig
from rag_engine.schema import Evidence, GeneratedAnswer


class AnswerGenerator(Protocol):
    def generate(self, query: str, evidence: list[Evidence]) -> GeneratedAnswer: ...


class MockGroundedGenerator:
    """Deterministic interface demo; not a capable natural-language model."""
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    def generate(self, query: str, evidence: list[Evidence]) -> GeneratedAnswer:
        eligible = [item for item in evidence if item.score >= self.config.minimum_evidence_score]
        if not eligible:
            if self.config.abstain_when_empty:
                return GeneratedAnswer("I do not have enough retrieved evidence to answer this question.", [], True, "insufficient_evidence")
            return GeneratedAnswer("No evidence was selected.", [], False, "empty_context")
        # The mock uses the first source excerpt so its citation is explicit and auditable.
        chosen = eligible[0]
        excerpt = re.sub(r"\s+", " ", chosen.text).strip()
        if len(excerpt) > 360:
            excerpt = excerpt[:357].rsplit(" ", 1)[0] + "..."
        answer = f"Relevant source excerpt: {excerpt}"
        return GeneratedAnswer(answer, [chosen.chunk_id], False, None)
