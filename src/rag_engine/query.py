"""Query rewriting contracts with an offline identity default."""
from __future__ import annotations

from typing import Protocol


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> list[str]: ...


class IdentityQueryRewriter:
    def rewrite(self, query: str) -> list[str]:
        cleaned = query.strip()
        return [cleaned] if cleaned else []


class StaticMultiQueryRewriter:
    """Useful for controlled experiments; strings are supplied by the caller."""
    def __init__(self, variants: dict[str, list[str]]) -> None:
        self.variants = {key.casefold(): [item.strip() for item in values if item.strip()] for key, values in variants.items()}

    def rewrite(self, query: str) -> list[str]:
        base = query.strip()
        if not base:
            return []
        candidates = [base, *self.variants.get(base.casefold(), [])]
        return list(dict.fromkeys(candidates))
