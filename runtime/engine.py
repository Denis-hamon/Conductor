"""Workflow execution engine — sequential and parallel step execution."""

import asyncio
import json
import logging
import time
from typing import Any

from runtime.models import validate_plan, ValidationError

logger = logging.getLogger("runtime.engine")


class StepResult:
    def __init__(self, step_index: int, step_type: str, agent: str, status: str,
                 latency_ms: float, output: Any = None, error: str = ""):
        self.step_index = step_index
        self.step_type = step_type
        self.agent = agent
        self.status = status
        self.latency_ms = latency_ms
        self.output = output
        self.error = error

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "type": self.step_type,
            "agent": self.agent,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "output": self.output,
            "error": self.error,
        }


class WorkflowEngine:
    def __init__(self, model_caller=None):
        self.model_caller = model_caller or self._default_model_call
        self.timeout = 30

    async def execute(self, plan: dict) -> dict:
        try:
            validate_plan(plan)
        except ValidationError as e:
            return {"status": "failed", "error": str(e), "steps": [], "latency_ms": 0}

        state = {"plan": plan, "step_results": [], "current_step": 0}
        start = time.monotonic()

        for i, step in enumerate(plan["steps"]):
            state["current_step"] = i
            step_type = step.get("type")

            if step_type == "parallel":
                result = await self._execute_parallel(step, i)
            elif step_type == "human_approval":
                result = await self._execute_human_approval(step, i)
            else:
                result = await self._execute_step(step, i)

            state["step_results"].append(result.to_dict())

            if result.status == "failed":
                break

        total_latency = (time.monotonic() - start) * 1000

        return {
            "status": "completed" if all(s["status"] == "completed" for s in state["step_results"]) else "failed",
            "steps": state["step_results"],
            "latency_ms": round(total_latency, 2),
        }

    async def _execute_step(self, step: dict, index: int) -> StepResult:
        agent = step.get("agent", "unknown")
        step_type = step.get("type", "call_agent")
        budget = step.get("budget_tokens", 2048)
        step_start = time.monotonic()

        try:
            output = await asyncio.wait_for(
                self.model_caller(agent, step_type, budget),
                timeout=self.timeout,
            )
            latency = (time.monotonic() - step_start) * 1000
            return StepResult(index, step_type, agent, "completed", round(latency, 2), output)
        except asyncio.TimeoutError:
            latency = (time.monotonic() - step_start) * 1000
            logger.warning("Step %d (%s) timed out after %.0fms", index, agent, latency)
            return StepResult(index, step_type, agent, "failed", round(latency, 2), error="timeout")
        except Exception as e:
            latency = (time.monotonic() - step_start) * 1000
            logger.error("Step %d (%s) failed: %s", index, agent, e)
            return StepResult(index, step_type, agent, "failed", round(latency, 2), error=str(e))

    async def _execute_parallel(self, step: dict, index: int) -> StepResult:
        agents = step.get("agents", [])
        if not agents:
            return StepResult(index, "parallel", "none", "failed", 0, error="no agents in parallel block")

        merge_strategy = step.get("merge_strategy", "evidence_weighted")
        step_start = time.monotonic()

        tasks = [self.model_caller(a, "call_agent", 2048) for a in agents]
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=self.timeout)
            latency = (time.monotonic() - step_start) * 1000
            outputs = [r for r in results if not isinstance(r, Exception)]
            errors = [str(r) for r in results if isinstance(r, Exception)]

            merged = self._merge_outputs(outputs, merge_strategy)
            status = "completed" if outputs else "failed"
            return StepResult(index, "parallel", ",".join(agents), status, round(latency, 2), merged,
                              error="; ".join(errors))
        except asyncio.TimeoutError:
            latency = (time.monotonic() - step_start) * 1000
            return StepResult(index, "parallel", ",".join(agents), "failed", round(latency, 2), error="timeout")

    def _merge_outputs(self, outputs: list, strategy: str) -> Any:
        if not outputs:
            return None
        if strategy == "evidence_weighted" and len(outputs) > 1:
            return {"merged": True, "strategy": strategy, "responses": outputs, "primary": outputs[0]}
        return outputs[0] if outputs else None

    async def _execute_human_approval(self, step: dict, index: int) -> StepResult:
        reason = step.get("reason", "unknown")
        logger.info("HITL pause: %s — awaiting approval webhook", reason)
        return StepResult(index, "human_approval", "human", "awaiting_approval", 0,
                          output={"reason": reason, "status": "awaiting_approval"})

    async def _default_model_call(self, agent: str, step_type: str, budget: int) -> dict:
        await asyncio.sleep(0.1)
        return {"agent": agent, "type": step_type, "content": f"Simulated response from {agent}"}
