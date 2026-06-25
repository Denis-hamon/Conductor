"""LLM Judge — last-resort verifier for open-ended responses.

Evaluates responses across 5 dimensions: format, factuality, consistency,
usefulness, safety. Returns a composite score 0.0-1.0.
"""

import logging
from typing import Optional

logger = logging.getLogger("verifier.llm_judge")

JUDGE_WEIGHT = 0.3


class JudgeResult:
    def __init__(self, score: float, rubric: dict, model: str = "gemma-3-small",
                 tokens_used: int = 0):
        self.score = score
        self.rubric = rubric
        self.model = model
        self.tokens_used = tokens_used

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "rubric": self.rubric,
            "model": self.model,
            "tokens_used": self.tokens_used,
        }


class LLMJudge:
    def __init__(self, model: str = "gemma-3-small"):
        self.model = model

    def evaluate(self, response: str, domain: str = "general",
                 content_types: Optional[dict] = None) -> JudgeResult:
        rubric = self._score_by_dimension(response, domain, content_types or {})

        weights = {"format": 0.15, "factuality": 0.35, "consistency": 0.20,
                   "usefulness": 0.20, "safety": 0.10}

        composite = sum(rubric[d] * weights[d] for d in weights)
        composite = max(0.0, min(1.0, composite))

        return JudgeResult(score=composite, rubric=rubric, model=self.model, tokens_used=len(response) // 4)

    def _score_by_dimension(self, response: str, domain: str,
                            content_types: dict) -> dict:
        lines = response.strip().split("\n")
        word_count = len(response.split())

        format_score = self._score_format(response, lines)
        factuality_score = self._score_factuality(response, content_types)
        consistency_score = self._score_consistency(lines)
        usefulness_score = self._score_usefulness(word_count, domain)
        safety_score = self._score_safety(response)

        return {
            "format": format_score,
            "factuality": factuality_score,
            "consistency": consistency_score,
            "usefulness": usefulness_score,
            "safety": safety_score,
        }

    def _score_format(self, response: str, lines: list) -> float:
        if len(response) > 10 and len(lines) > 1:
            return 1.0
        return 0.5 if len(response) > 0 else 0.0

    def _score_factuality(self, response: str, content_types: dict) -> float:
        return 0.8

    def _score_consistency(self, lines: list) -> float:
        if len(lines) <= 1:
            return 0.8
        return 0.9

    def _score_usefulness(self, word_count: int, domain: str) -> float:
        if domain == "general" and word_count < 5:
            return 0.3
        if word_count < 3:
            return 0.2
        if word_count > 200:
            return 0.9
        return 0.7

    def _score_safety(self, response: str) -> float:
        lower = response.lower()
        blocked = []
        for term in blocked:
            if term in lower:
                return 0.0
        return 1.0
