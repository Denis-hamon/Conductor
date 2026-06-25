"""RAG citation verifier — checks each assertion is backed by a source citation."""

import logging
import re
from typing import Optional

logger = logging.getLogger("verifier.rag")

CITATION_PATTERN = re.compile(r'\[source:\s*([^,\]]+)(?:,\s*passage:\s*"([^"]*)")?\]')


class RAGResult:
    def __init__(self, score: float, verified: int = 0, total: int = 0,
                 unverified: list[str] = None, na: bool = False):
        self.score = score
        self.verified = verified
        self.total = total
        self.unverified = unverified or []
        self.na = na

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "verified": self.verified,
            "total": self.total,
            "unverified": self.unverified,
            "na": self.na,
        }


class RAGVerifier:
    def verify(self, response: str, sources: Optional[list[str]] = None) -> RAGResult:
        if not sources:
            return RAGResult(score=0.0, na=True)

        citations = CITATION_PATTERN.findall(response)

        if not citations:
            return RAGResult(score=0.0, total=1, unverified=["no citations found in response"], na=True)

        verified_sources = {c[0] for c in citations}
        valid_citations = [c for c in citations if c[0] in sources]

        verified_count = len(valid_citations)
        total_assertions = len(citations)
        score = verified_count / total_assertions if total_assertions > 0 else 0.0

        unverified = []
        for c in citations:
            if c[0] not in sources:
                unverified.append(f"citation source '{c[0]}' not in provided sources")

        return RAGResult(score=score, verified=verified_count, total=total_assertions,
                         unverified=unverified)
