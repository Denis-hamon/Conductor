"""Tests for WorkflowEngine — sequential and parallel step execution."""

import pytest

from runtime.engine import WorkflowEngine, StepResult


@pytest.fixture
def engine():
    return WorkflowEngine()


def make_plan(steps=None, verification_gates=None):
    return {
        "steps": steps or [
            {"type": "call_agent", "agent": "qwen", "role": "worker", "budget_tokens": 1024},
        ],
        "verification_gates": verification_gates or [],
        "stop_condition": "on_complete",
    }


@pytest.mark.asyncio
class TestWorkflowEngine:
    async def test_execute_single_step(self, engine):
        plan = make_plan()
        result = await engine.execute(plan)
        assert result["status"] == "completed"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["status"] == "completed"

    async def test_execute_multiple_steps(self, engine):
        plan = make_plan(steps=[
            {"type": "call_agent", "agent": "qwen", "role": "worker", "budget_tokens": 1024},
            {"type": "call_agent", "agent": "deepseek", "role": "worker", "budget_tokens": 2048},
        ])
        result = await engine.execute(plan)
        assert result["status"] == "completed"
        assert len(result["steps"]) == 2
        assert all(s["status"] == "completed" for s in result["steps"])

    async def test_invalid_plan_returns_failed(self, engine):
        result = await engine.execute({"invalid": True})
        assert result["status"] == "failed"
        assert "error" in result

    async def test_failed_step_stops_execution(self, engine):
        async def failing_model(agent, step_type, budget):
            raise RuntimeError("model unavailable")
        engine.model_caller = failing_model
        plan = make_plan(steps=[
            {"type": "call_agent", "agent": "broken", "role": "worker", "budget_tokens": 1024},
            {"type": "call_agent", "agent": "qwen", "role": "worker", "budget_tokens": 1024},
        ])
        result = await engine.execute(plan)
        assert result["status"] == "failed"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["status"] == "failed"

    async def test_step_timeout_returns_failed(self, engine):
        async def slow_model(agent, step_type, budget):
            import asyncio
            await asyncio.sleep(100)
        engine.model_caller = slow_model
        engine.timeout = 0.05
        plan = make_plan()
        result = await engine.execute(plan)
        assert result["status"] == "failed"
        assert "timeout" in result["steps"][0]["error"]

    async def test_parallel_execution(self, engine):
        plan = make_plan(steps=[
            {
                "type": "parallel",
                "agent": "ensemble",
                "role": "worker",
                "agents": ["qwen-coder", "deepseek-reasoner"],
                "merge_strategy": "evidence_weighted",
            },
        ])
        result = await engine.execute(plan)
        assert result["status"] == "completed"
        assert result["steps"][0]["type"] == "parallel"

    async def test_parallel_with_no_agents_fails(self, engine):
        plan = make_plan(steps=[
            {"type": "parallel", "agent": "ensemble", "role": "worker", "agents": []},
        ])
        result = await engine.execute(plan)
        assert result["steps"][0]["status"] == "failed"
        assert "no agents" in result["steps"][0]["error"]

    async def test_human_approval_step_returns_awaiting(self, engine):
        plan = make_plan(steps=[
            {"type": "call_agent", "agent": "qwen", "role": "worker", "budget_tokens": 1024},
            {"type": "human_approval", "agent": "human", "role": "approver", "reason": "Deploy to prod?"},
        ])
        result = await engine.execute(plan)
        assert result["steps"][1]["status"] == "awaiting_approval"

    async def test_returns_latency_ms(self, engine):
        plan = make_plan()
        result = await engine.execute(plan)
        assert result["latency_ms"] > 0

    async def test_step_result_to_dict(self):
        step = StepResult(0, "call_agent", "qwen", "completed", 150.5, {"ok": True})
        d = step.to_dict()
        assert d["step_index"] == 0
        assert d["type"] == "call_agent"
        assert d["agent"] == "qwen"
        assert d["status"] == "completed"
        assert d["latency_ms"] == 150.5
        assert d["output"] == {"ok": True}

    async def test_step_result_with_error(self):
        step = StepResult(1, "call_agent", "qwen", "failed", 200.0, error="timeout")
        d = step.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "timeout"
