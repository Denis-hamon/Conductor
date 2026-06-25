"""ATDD: Reward calculation tests (R-11, V-13).

AD-6: Traces as the single source of truth for audit and training
"""

import pytest


def calculate_reward(verifiers: list[dict]) -> float:
    """Calculate weighted Reward from verifier scores."""
    weighted_sum = 0.0
    total_weight = 0.0
    for v in verifiers:
        if v.get("type") == "N/A":
            continue
        weight = v["weight"]
        if v["type"] == "llm_judge" and weight > 0.3:
            weight = 0.3
        weighted_sum += v["score"] * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return round(weighted_sum / total_weight, 4)


class TestRewardCalculation:
    """R-11 / V-13: Reward composition."""

    @pytest.mark.atdd
    def test_sandbox_only_verification(self):
        """Sandbox-only → Reward = sandbox score."""
        verifiers = [
            {"id": "sandbox-1", "type": "sandbox", "score": 0.8, "weight": 1.0},
        ]
        reward = calculate_reward(verifiers)
        assert reward == 0.8

    @pytest.mark.atdd
    def test_mixed_verification_with_weights(self):
        """Multiple verifiers with different weights."""
        verifiers = [
            {"id": "sandbox-1", "type": "sandbox", "score": 0.9, "weight": 0.4},
            {"id": "rag-1", "type": "rag", "score": 0.7, "weight": 0.4},
            {"id": "judge-1", "type": "llm_judge", "score": 0.6, "weight": 0.2},
        ]
        expected = (0.9 * 0.4 + 0.7 * 0.4 + 0.6 * 0.2) / 1.0
        assert calculate_reward(verifiers) == pytest.approx(expected)

    @pytest.mark.atdd
    def test_llm_judge_weight_cap(self):
        """LLM judge weight > 0.3 → capped at 0.3."""
        verifiers = [
            {"id": "sandbox-1", "type": "sandbox", "score": 0.8, "weight": 0.7},
            {"id": "judge-1", "type": "llm_judge", "score": 0.6, "weight": 0.5},
        ]
        reward = calculate_reward(verifiers)
        capped_weight = min(0.5, 0.3)
        expected = (0.8 * 0.7 + 0.6 * capped_weight) / (0.7 + capped_weight)
        assert reward == pytest.approx(expected)

    @pytest.mark.atdd
    def test_verifier_skipped_na(self):
        """RAG verifier N/A → excluded, remaining renormalized."""
        verifiers = [
            {"id": "sandbox-1", "type": "sandbox", "score": 0.85, "weight": 0.6},
            {"id": "rag-1", "type": "N/A", "score": 0.0, "weight": 0.4},
            {"id": "judge-1", "type": "llm_judge", "score": 0.7, "weight": 0.4},
        ]
        capped_judge = min(0.4, 0.3)
        expected = (0.85 * 0.6 + 0.7 * capped_judge) / (0.6 + capped_judge)
        assert calculate_reward(verifiers) == pytest.approx(expected)

    @pytest.mark.atdd
    def test_all_verifiers_fail(self):
        """All verifiers return 0 → Reward = 0.0."""
        verifiers = [
            {"id": "sandbox-1", "type": "sandbox", "score": 0.0, "weight": 0.5},
            {"id": "rag-1", "type": "rag", "score": 0.0, "weight": 0.5},
        ]
        assert calculate_reward(verifiers) == 0.0
