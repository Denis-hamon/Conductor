"""Deterministic fallback planner — used when Conductor fails."""

import logging

logger = logging.getLogger("conductor.fallback")


class FallbackPlanner:
    def generate(self, content: str) -> dict:
        logger.warning("Fallback planner activated for: %.60s", content)
        return {
            "plan_id": id(content),
            "domain": "general",
            "fallback": True,
            "steps": [
                {"type": "call_agent", "agent": "qwen-reasoner", "role": "Worker", "budget_tokens": 2048},
            ],
            "verification_gates": [
                {"type": "llm_judge", "weight": 0.3},
            ],
            "stop_condition": "on_complete",
        }
