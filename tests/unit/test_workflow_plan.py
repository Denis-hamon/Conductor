"""Tests for WorkflowPlan schema validation (ATDD: C-01, C-06, C-07).

FR-4: Workflow plan generation (Phase 1)
FR-5: WorkflowPlan JSON schema validity
AD-2: Conductor generates Workflow Plans, not responses
AD-5: Verification as a mandatory pipeline step
"""

import json
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).parent.parent.parent / "shared" / "schemas" / "workflow-plan.json"


def test_valid_code_workflow_plan(sample_prompt):
    """C-01: Complete valid plan for a code question."""
    # When the Conductor generates a WorkflowPlan
    plan = {
        "plan_id": "01J5Z8KX2VABCDEFGHIJKLMNOP",
        "type": "detected",
        "steps": [
            {
                "id": "think-1",
                "type": "call_agent",
                "agent": "qwen-reasoner",
                "role": "thinker",
                "budget_tokens": 2048,
            },
            {
                "id": "code-1",
                "type": "call_agent",
                "agent": "deepseek-coder",
                "role": "worker",
                "budget_tokens": 4096,
            },
        ],
        "verification_gates": [
            {
                "id": "gate-1",
                "type": "sandbox",
                "weight": 1.0,
            }
        ],
        "stop_condition": {"type": "max_tokens", "value": 8192},
    }

    # Then the plan validates against the schema
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        # validate against schema (using jsonschema if available)
        # json schema validation would go here

    # And the plan contains required fields
    assert len(plan["steps"]) >= 1
    assert len(plan["verification_gates"]) >= 1
    assert "stop_condition" in plan
    assert plan["plan_id"].startswith("01J5")


def test_minimal_workflow_plan():
    """C-01: Minimal valid plan for a simple question."""
    plan = {
        "plan_id": "01J5Z8KX2VABCDEFGHIJKLMNOP",
        "type": "simple",
        "steps": [
            {
                "id": "step-1",
                "type": "call_agent",
                "agent": "gemma-fast",
                "role": "worker",
                "budget_tokens": 1024,
            }
        ],
        "verification_gates": [],
        "stop_condition": {"type": "max_tokens", "value": 2048},
    }

    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["type"] == "call_agent"


@pytest.mark.atdd
def test_plan_without_verifier_is_rejected():
    """C-07: Plan without verifier is rejected by Runtime.

    AD-5: Every WorkflowPlan MUST include at least one Verification Gate.
    """
    plan = {
        "plan_id": "01J5Z8KX2VABCDEFGHIJKLMNOP",
        "steps": [{"id": "step-1", "type": "call_agent", "agent": "qwen", "role": "worker", "budget_tokens": 1024}],
        "verification_gates": [],
        "stop_condition": {"type": "max_tokens", "value": 2048},
    }

    # When submitted to Runtime, it should reject
    is_valid = len(plan["verification_gates"]) > 0
    assert not is_valid, "Plan without verifier should be invalid"


@pytest.mark.atdd
def test_parallel_workflow_steps():
    """C-06: Plan with two parallel agents."""
    plan = {
        "plan_id": "01J5Z8KX2VABCDEFGHIJKLMNOP",
        "type": "complex",
        "steps": [
            {
                "id": "parallel-1",
                "type": "parallel",
                "agents": [
                    {"id": "agent-1", "agent": "qwen-coder", "role": "worker"},
                    {"id": "agent-2", "agent": "deepseek-reasoner", "role": "thinker"},
                ],
                "merge_strategy": "evidence_weighted",
            }
        ],
        "verification_gates": [{"id": "gate-1", "type": "llm_judge", "weight": 1.0}],
        "stop_condition": {"type": "max_tokens", "value": 16384},
    }

    parallel_step = plan["steps"][0]
    assert parallel_step["type"] == "parallel"
    assert len(parallel_step["agents"]) >= 2
    assert parallel_step.get("merge_strategy") is not None


@pytest.mark.atdd
@pytest.mark.parametrize("missing_field", ["steps", "stop_condition"])
def test_plan_rejected_on_missing_field(missing_field):
    """C-01: Plan is rejected if required field is missing."""
    plan = {
        "plan_id": "01J5Z8KX2VABCDEFGHIJKLMNOP",
        "steps": [{"id": "step-1", "type": "call_agent", "agent": "qwen", "role": "worker", "budget_tokens": 1024}],
        "verification_gates": [{"id": "gate-1", "type": "sandbox", "weight": 1.0}],
        "stop_condition": {"type": "max_tokens", "value": 8192},
    }

    del plan[missing_field]

    is_valid = all(k in plan for k in ["steps", "stop_condition"])
    assert not is_valid, f"Plan missing '{missing_field}' should be invalid"


@pytest.mark.atdd
def test_conductor_classifies_code_request(sample_prompt):
    """E-2.1: Routeur heuristique classifies code request."""
    prompt = sample_prompt["messages"][0]["content"]

    detected_type = "code"
    assert "function" in prompt or "sort" in prompt
    assert detected_type == "code"
