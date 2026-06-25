"""Tests for plan validation models — validate_plan()."""

import pytest

from runtime.models import validate_plan, ValidationError


def valid_plan():
    return {
        "steps": [
            {"type": "call_agent", "agent": "qwen", "role": "worker", "budget_tokens": 1024},
        ],
        "verification_gates": [
            {"type": "sandbox", "weight": 1.0},
        ],
        "stop_condition": "on_complete",
    }


class TestValidatePlan:
    def test_valid_plan_passes(self):
        plan = valid_plan()
        validate_plan(plan)

    def test_not_a_dict_raises(self):
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_plan("not a dict")

    def test_missing_steps_raises(self):
        plan = valid_plan()
        del plan["steps"]
        with pytest.raises(ValidationError, match="steps must be a non-empty array"):
            validate_plan(plan)

    def test_empty_steps_raises(self):
        plan = valid_plan()
        plan["steps"] = []
        with pytest.raises(ValidationError, match="steps must be a non-empty array"):
            validate_plan(plan)

    def test_steps_not_a_list_raises(self):
        plan = valid_plan()
        plan["steps"] = "not a list"
        with pytest.raises(ValidationError, match="steps must be a non-empty array"):
            validate_plan(plan)

    def test_missing_verification_gates_raises(self):
        plan = valid_plan()
        del plan["verification_gates"]
        with pytest.raises(ValidationError, match="verification_gates must be an array"):
            validate_plan(plan)

    def test_verification_gates_not_list_raises(self):
        plan = valid_plan()
        plan["verification_gates"] = "not a list"
        with pytest.raises(ValidationError, match="verification_gates must be an array"):
            validate_plan(plan)

    def test_missing_stop_condition_raises(self):
        plan = valid_plan()
        del plan["stop_condition"]
        with pytest.raises(ValidationError, match="stop_condition must be a string"):
            validate_plan(plan)

    def test_stop_condition_not_string_raises(self):
        plan = valid_plan()
        plan["stop_condition"] = 42
        with pytest.raises(ValidationError, match="stop_condition must be a string"):
            validate_plan(plan)

    def test_step_not_a_dict_raises(self):
        plan = valid_plan()
        plan["steps"].append("not a dict")
        with pytest.raises(ValidationError, match="step 1 must be an object"):
            validate_plan(plan)

    def test_step_invalid_type_raises(self):
        plan = valid_plan()
        plan["steps"].append({"type": "invalid_type", "agent": "qwen", "role": "worker"})
        with pytest.raises(ValidationError, match="unknown type"):
            validate_plan(plan)

    def test_step_missing_agent_raises(self):
        plan = valid_plan()
        plan["steps"].append({"type": "call_agent", "role": "worker"})
        with pytest.raises(ValidationError, match="missing 'agent'"):
            validate_plan(plan)

    def test_step_missing_role_raises(self):
        plan = valid_plan()
        plan["steps"].append({"type": "call_agent", "agent": "qwen"})
        with pytest.raises(ValidationError, match="missing 'role'"):
            validate_plan(plan)

    def test_all_valid_step_types_pass(self):
        for step_type in ["call_agent", "parallel", "verify", "merge", "human_approval"]:
            plan = valid_plan()
            plan["steps"].append({"type": step_type, "agent": "qwen", "role": "worker"})
            validate_plan(plan)

    def test_empty_verification_gates_is_valid(self):
        plan = valid_plan()
        plan["verification_gates"] = []
        validate_plan(plan)

    def test_plan_with_parallel_step_is_valid(self):
        plan = valid_plan()
        plan["steps"] = [
            {"type": "parallel", "agent": "ensemble", "role": "worker",
             "agents": [{"id": "a1"}, {"id": "a2"}]},
        ]
        validate_plan(plan)

    def test_plan_with_human_approval_is_valid(self):
        plan = valid_plan()
        plan["steps"].append({"type": "human_approval", "agent": "human", "role": "approver"})
        validate_plan(plan)
