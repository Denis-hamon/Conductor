"""Tests for WorkflowPlanner — plan generation per domain."""

import pytest

from conductor.planner import WorkflowPlanner, DOMAIN_CONFIG


@pytest.fixture
def planner():
    return WorkflowPlanner()


class TestWorkflowPlanner:
    def test_generates_code_plan(self, planner):
        plan = planner.generate("code", "Write a Python function to sort a list")
        assert plan["domain"] == "code"
        assert len(plan["steps"]) >= 2
        assert any(s["template"] == "code" for s in plan["steps"] if "template" in s)
        assert len(plan["verification_gates"]) >= 1
        assert plan["verification_gates"][0]["type"] == "sandbox_code"

    def test_generates_rag_plan(self, planner):
        plan = planner.generate("rag", "Summarize the contract")
        assert plan["domain"] == "rag"
        assert plan["verification_gates"][0]["type"] == "rag_citation"

    def test_generates_reason_plan(self, planner):
        plan = planner.generate("reason", "Explain why the sky is blue")
        assert plan["domain"] == "reason"
        assert plan["verification_gates"][0]["type"] == "llm_judge"

    def test_generates_mcp_plan(self, planner):
        plan = planner.generate("mcp", "Call the weather API")
        assert plan["domain"] == "mcp"
        assert plan["verification_gates"][0]["type"] == "simulated"

    def test_generates_general_plan(self, planner):
        plan = planner.generate("general", "Hello, how are you?")
        assert plan["domain"] == "general"
        assert plan["verification_gates"][0]["type"] == "llm_judge"

    def test_unknown_domain_passes_through(self, planner):
        plan = planner.generate("unknown_domain", "Anything")
        assert plan["domain"] == "unknown_domain"

    def test_plan_has_required_structure(self, planner):
        plan = planner.generate("code", "Write code")
        assert "plan_id" in plan
        assert "domain" in plan
        assert "steps" in plan
        assert "verification_gates" in plan
        assert "stop_condition" in plan

    def test_first_step_is_always_conductor_thinker(self, planner):
        plan = planner.generate("code", "Write code")
        assert plan["steps"][0]["agent"] == "conductor-thinker"
        assert plan["steps"][0]["role"] == "Thinker"

    def test_second_step_uses_domain_model(self, planner):
        plan = planner.generate("rag", "Search document")
        assert plan["steps"][1]["agent"] == DOMAIN_CONFIG["rag"]["model"]

    def test_verification_weight_matches_domain(self, planner):
        plan = planner.generate("code", "Write code")
        assert plan["verification_gates"][0]["weight"] == DOMAIN_CONFIG["code"]["verifier_weight"]

    def test_generate_uses_content_for_plan_id(self, planner):
        plan_a = planner.generate("general", "Hello")
        plan_b = planner.generate("general", "World")
        assert plan_a["plan_id"] != plan_b["plan_id"]

    def test_code_plan_has_sandbox_verifier(self, planner):
        plan = planner.generate("code", "Write code")
        types = [g["type"] for g in plan["verification_gates"]]
        assert "sandbox_code" in types
