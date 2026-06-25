"""Tests for CPT Phase 0 — data formatting and synthetic generation."""

import json
import tempfile
from pathlib import Path

import pytest

from training.phase0_cpt.data.format_sequences import (
    format_example,
    TrainingExample,
    TEMPLATE,
)
from training.phase0_cpt.data.generate_synthetic import (
    generate_trace,
    generate_dataset,
)


class TestFormatSequences:
    def test_format_example_with_valid_trace(self):
        trace = {
            "request": {
                "model": "test",
                "messages": [{"role": "user", "content": "Write a Python function"}],
            },
            "workflow_plan": {"domain": "code", "steps": [], "verification_gates": [], "stop_condition": "on_complete"},
            "steps": [{"step_id": "s1", "type": "call_agent", "agent": "qwen-coder", "output": "done"}],
            "rubric": {"reward": 0.85},
            "metadata": {"domain": "code", "difficulty": "medium"},
        }

        example = format_example(trace)
        assert example is not None
        assert example.query == "Write a Python function"
        assert example.plan["domain"] == "code"
        assert example.reward == 0.85
        assert example.domain == "code"
        assert isinstance(example.sequence, str)
        assert "Write a Python function" in example.sequence
        assert "0.85" in example.sequence

    def test_format_example_missing_request(self):
        trace = {"workflow_plan": {}, "steps": []}
        example = format_example(trace)
        assert example is None

    def test_format_example_empty_messages(self):
        trace = {
            "request": {"model": "test", "messages": []},
            "workflow_plan": {"domain": "general"},
            "rubric": {"reward": 0.0},
        }
        example = format_example(trace)
        assert example is not None
        assert example.query == ""

    def test_format_example_string_request(self):
        trace = {
            "request": "Write code",
            "workflow_plan": {"domain": "code"},
        }
        example = format_example(trace)
        assert example is not None
        assert example.query == "Write code"

    def test_format_example_json_string_fields(self):
        trace = {
            "request": json.dumps({"model": "test", "messages": [{"role": "user", "content": "Hello from JSON string"}]}),
            "workflow_plan": json.dumps({"domain": "code", "steps": [], "verification_gates": [], "stop_condition": "on_complete"}),
            "steps": json.dumps([{"step_id": "s1", "type": "call_agent"}]),
            "rubric": json.dumps({"reward": 0.75}),
            "metadata": json.dumps({"domain": "code", "difficulty": "medium"}),
        }
        example = format_example(trace)
        assert example is not None
        assert example.query == "Hello from JSON string"
        assert example.plan["domain"] == "code"
        assert example.reward == 0.75
        assert example.domain == "code"


class TestSyntheticGeneration:
    def test_generate_trace_structure(self):
        trace = generate_trace("code", "medium", 1, 1000000)
        assert trace["trace_id"] == "syn_tr_000001"
        assert trace["workflow_id"] == "syn_wf_000001"
        assert trace["metadata"]["domain"] == "code"
        assert trace["metadata"]["difficulty"] == "medium"
        assert trace["metadata"]["source"] == "synthetic"
        assert "messages" in trace["request"]
        assert "steps" in trace["workflow_plan"]
        assert "steps" in trace
        assert trace["reward"] > 0

    def test_generate_trace_domain_variants(self):
        for domain in ["code", "rag", "reason", "general", "mcp"]:
            trace = generate_trace(domain, "easy", 0, 1000000)
            assert trace["metadata"]["domain"] == domain

    def test_generate_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = generate_dataset(tmp, num_traces=100, seed=42)
            path = Path(output)
            assert path.exists()
            assert path.suffix == ".parquet"
            assert path.stat().st_size > 0

            meta_path = Path(tmp) / "generation_meta.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta["num_traces"] == 100


class TestSequenceTemplate:
    def test_template_contains_all_sections(self):
        example = TrainingExample(
            query="test query",
            plan={"domain": "code"},
            steps=[{"id": "s1", "type": "call_agent"}],
            reward=0.9,
            domain="code",
            difficulty="easy",
        )
        result = example.sequence or TEMPLATE.format(
            query=example.query,
            plan_json=json.dumps(example.plan),
            steps_json=json.dumps(example.steps),
            reward=example.reward,
            domain=example.domain,
        )
        assert "### Request" in result
        assert "### Plan" in result
        assert "### Steps" in result
        assert "### Reward" in result
        assert "### Domain" in result
        assert "test query" in result
        assert "0.9" in result
