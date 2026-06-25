"""Tests for FallbackPlanner — deterministic fallback plan generation."""

import pytest

from conductor.fallback import FallbackPlanner


@pytest.fixture
def fallback():
    return FallbackPlanner()


class TestFallbackPlanner:
    def test_generates_general_domain(self, fallback):
        plan = fallback.generate("anything")
        assert plan["domain"] == "general"

    def test_marks_as_fallback(self, fallback):
        plan = fallback.generate("anything")
        assert plan["fallback"] is True

    def test_has_single_step(self, fallback):
        plan = fallback.generate("anything")
        assert len(plan["steps"]) == 1

    def test_step_uses_qwen_reasoner(self, fallback):
        plan = fallback.generate("anything")
        assert plan["steps"][0]["agent"] == "qwen-reasoner"

    def test_has_llm_judge_verifier(self, fallback):
        plan = fallback.generate("anything")
        assert len(plan["verification_gates"]) == 1
        assert plan["verification_gates"][0]["type"] == "llm_judge"

    def test_has_stop_condition(self, fallback):
        plan = fallback.generate("anything")
        assert plan["stop_condition"] == "on_complete"

    def test_plan_id_varies_by_content(self, fallback):
        plan_a = fallback.generate("hello")
        plan_b = fallback.generate("world")
        assert plan_a["plan_id"] != plan_b["plan_id"]

    def test_has_required_fields(self, fallback):
        plan = fallback.generate("anything")
        assert "plan_id" in plan
        assert "domain" in plan
        assert "fallback" in plan
        assert "steps" in plan
        assert "verification_gates" in plan
        assert "stop_condition" in plan

    def test_verifier_weight_is_low(self, fallback):
        plan = fallback.generate("anything")
        assert plan["verification_gates"][0]["weight"] == 0.3

    @pytest.mark.atdd
    def test_fallback_activated_on_empty_content(self, fallback):
        plan = fallback.generate("")
        assert plan["fallback"] is True
        assert len(plan["steps"]) == 1
