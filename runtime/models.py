"""Workflow plan data models and validation."""

import json
from typing import Any


class ValidationError(Exception):
    pass


def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict):
        raise ValidationError("plan must be a JSON object")

    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValidationError("invalid plan: steps must be a non-empty array")

    gates = plan.get("verification_gates")
    if not isinstance(gates, list):
        raise ValidationError("invalid plan: verification_gates must be an array")

    stop = plan.get("stop_condition")
    if not isinstance(stop, str):
        raise ValidationError("invalid plan: stop_condition must be a string")

    valid_types = {"call_agent", "parallel", "verify", "merge", "human_approval"}
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValidationError(f"invalid plan: step {i} must be an object")
        if step.get("type") not in valid_types:
            raise ValidationError(f"invalid plan: step {i} has unknown type '{step.get('type')}'")
        if "agent" not in step:
            raise ValidationError(f"invalid plan: step {i} missing 'agent'")
        if "role" not in step:
            raise ValidationError(f"invalid plan: step {i} missing 'role'")
