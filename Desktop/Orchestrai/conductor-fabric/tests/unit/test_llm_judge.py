"""Tests for LLMJudge — rubric-based response evaluation."""

import importlib.util
import os
import pytest

_judge_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "runtime", "internal", "verifier", "llm-judge", "judge.py"
)
_spec = importlib.util.spec_from_file_location("judge", _judge_path)
_judge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_judge)
LLMJudge = _judge.LLMJudge
JudgeResult = _judge.JudgeResult


@pytest.fixture
def judge():
    return LLMJudge()


class TestLLMJudge:
    def test_well_formatted_response_scores_high(self, judge):
        result = judge.evaluate("Line one\nLine two\nLine three", domain="general")
        assert result.score >= 0.7
        assert result.rubric["format"] == 1.0

    def test_empty_response_scores_format_zero(self, judge):
        result = judge.evaluate("", domain="general")
        assert result.rubric["format"] == 0.0

    def test_short_response_scores_lower_usefulness(self, judge):
        result = judge.evaluate("Hi", domain="general")
        assert result.rubric["usefulness"] == 0.3

    def test_long_response_scores_higher_usefulness(self, judge):
        result = judge.evaluate(" ".join(["word"] * 250), domain="general")
        assert result.rubric["usefulness"] == 0.9

    def test_code_domain_with_very_short_response(self, judge):
        result = judge.evaluate("Hi", domain="code")
        assert result.rubric["usefulness"] == 0.2

    def test_general_domain_short_response_usefulness(self, judge):
        result = judge.evaluate("ab", domain="general")
        assert result.rubric["usefulness"] == 0.3

    def test_score_is_between_0_and_1(self, judge):
        result = judge.evaluate("A" * 1000, domain="general")
        assert 0.0 <= result.score <= 1.0

    def test_score_uses_weights(self, judge):
        result = judge.evaluate("Line one\nLine two", domain="general")
        weights = {"format": 0.15, "factuality": 0.35, "consistency": 0.20,
                   "usefulness": 0.20, "safety": 0.10}
        expected = sum(result.rubric[d] * weights[d] for d in weights)
        assert result.score == pytest.approx(expected)

    def test_tokens_used_is_approximate(self, judge):
        result = judge.evaluate("Hello world test", domain="general")
        assert result.tokens_used > 0

    def test_judge_result_to_dict(self):
        rubric = {"format": 1.0, "factuality": 0.8, "consistency": 0.9,
                  "usefulness": 0.7, "safety": 1.0}
        result = JudgeResult(score=0.85, rubric=rubric, model="gemma-3-small", tokens_used=50)
        d = result.to_dict()
        assert d["score"] == 0.85
        assert d["model"] == "gemma-3-small"
        assert d["tokens_used"] == 50
