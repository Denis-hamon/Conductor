"""Shared test configuration for Conductor Fabric tests."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

SCHEMA_DIR = Path(__file__).parent.parent.parent / "shared" / "schemas"


def load_json_schema(name: str) -> dict[str, Any]:
    """Load a JSON schema from shared/schemas/."""
    path = SCHEMA_DIR / name
    if not path.exists():
        pytest.skip(f"Schema not found: {path}")
    with open(path) as f:
        return json.load(f)


def valid_workflow_plan(**overrides: Any) -> dict[str, Any]:
    """Build a valid WorkflowPlan for testing.

    Usage:
        plan = valid_workflow_plan(steps=[...])
    """
    plan = {
        "plan_id": "01J5Z8KX2VABCDEFGHIJKLMNOP",
        "steps": [
            {
                "id": "step-1",
                "type": "call_agent",
                "agent": "qwen-reasoner",
                "role": "thinker",
                "budget_tokens": 4096,
            }
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
    plan.update(overrides)
    return plan


def valid_verification_result(**overrides: Any) -> dict[str, Any]:
    """Build a valid VerificationResult for testing."""
    result = {
        "verifier_id": "sandbox-1",
        "type": "sandbox",
        "score": 1.0,
        "evidence": [
            {"test": "test_add", "passed": True, "output": ""}
        ],
        "passed": True,
    }
    result.update(overrides)
    return result


@pytest.fixture
def sample_code_block() -> str:
    return """
def add(a, b):
    return a + b
"""


@pytest.fixture
def sample_unit_tests() -> str:
    return """
def test_add_positive():
    assert add(1, 2) == 3
def test_add_negative():
    assert add(-1, -2) == -3
"""


@pytest.fixture
def sample_prompt() -> dict[str, Any]:
    return {
        "model": "conductor-fabric",
        "messages": [
            {"role": "user", "content": "Write a Python function to sort a list"}
        ],
    }


@pytest.fixture
def benchmark_dataset() -> list[dict[str, Any]]:
    """Standardized 100-prompt benchmark dataset for SM-1/SM-2/SM-3."""
    return [
        {"id": f"bench-{i}", "category": "code", "prompt": f"Write a function to {task}"}
        for i, task in enumerate([
            "sort a list",
            "reverse a string",
            "find primes",
            "calculate fibonacci",
            "parse JSON",
        ], 1)
    ]
