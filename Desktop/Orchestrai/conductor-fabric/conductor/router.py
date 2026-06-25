"""Heuristic request classifier — routes requests by domain."""

import re
from typing import NamedTuple


class RouteResult(NamedTuple):
    domain: str
    confidence: float
    reason: str


CODE_PATTERNS = [
    r"write\s+(a\s+)?(python|javascript|go|rust|java|typescript|bash|sql)\s+",
    r"(implement|code|program|function|class|script|algorithm)\s",
    r"def\s+\w+\s*\(|fn\s+\w+|function\s+\w+",
    r"how\s+(to|do\s+I)\s+(code|write|implement|program)",
    r"(debug|compile|syntax|error|exception|bug|fix)",
]

RAG_PATTERNS = [
    r"(according\s+to|based\s+on|from\s+the\s+(document|source|file))",
    r"(summarize|extract|find|search|retrieve|look\s+up)\s",
    r"(contract|clause|article|section|legal|law|regulation|compliance)",
    r"(citation|source|reference|quote)",
]

REASON_PATTERNS = [
    r"(math|equation|calculate|compute|solve|proof|theorem|logic)",
    r"(reason|analyze|compare|contrast|evaluate|assess|deduce)",
    r"(why|how\s+does|explain\s+(the\s+)?(reason|logic|cause))",
]

MCP_PATTERNS = [
    r"(call\s+(the\s+)?(tool|function|api|endpoint|service))",
    r"(use\s+(the\s+)?(tool|integration|connector))",
    r"(execute\s+(a\s+)?(command|query|action))",
    r"(mcp|model\s+context\s+protocol)",
]


def _score_patterns(text: str, patterns: list[str]) -> float:
    text_lower = text.lower()
    matches = sum(1 for p in patterns if re.search(p, text_lower))
    return matches / len(patterns) if patterns else 0.0


def classify_request(content: str) -> RouteResult:
    scores = {
        "code": _score_patterns(content, CODE_PATTERNS),
        "rag": _score_patterns(content, RAG_PATTERNS),
        "reason": _score_patterns(content, REASON_PATTERNS),
        "mcp": _score_patterns(content, MCP_PATTERNS),
    }

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    if best_score >= 0.3:
        return RouteResult(domain=best_domain, confidence=best_score, reason=f"matched {best_domain} patterns")

    return RouteResult(domain="general", confidence=0.2, reason="no high-confidence domain match")
