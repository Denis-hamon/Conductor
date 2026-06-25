"""WorkflowPlan generator — produces plans from classified requests."""

import json
import os

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")

DOMAIN_CONFIG = {
    "code": {
        "model": "qwen-coder",
        "template": "code",
        "verification": ["sandbox_code"],
        "verifier_weight": 0.8,
        "type": "call_agent",
    },
    "rag": {
        "model": "qwen-reasoner",
        "template": "rag",
        "verification": ["rag_citation"],
        "verifier_weight": 0.7,
        "type": "call_agent",
    },
    "reason": {
        "model": "qwen-reasoner",
        "template": "reason",
        "verification": ["llm_judge"],
        "verifier_weight": 0.6,
        "type": "call_agent",
    },
    "general": {
        "model": "gemma-3-small",
        "template": "general",
        "verification": ["llm_judge"],
        "verifier_weight": 0.3,
        "type": "call_agent",
    },
    "mcp": {
        "model": "qwen-reasoner",
        "template": "mcp",
        "verification": ["simulated"],
        "verifier_weight": 0.5,
        "type": "call_agent",
    },
}


class WorkflowPlanner:
    def generate(self, domain: str, content: str) -> dict:
        cfg = DOMAIN_CONFIG.get(domain, DOMAIN_CONFIG["general"])

        plan = {
            "plan_id": id(content),
            "domain": domain,
            "steps": [],
            "verification_gates": [],
            "stop_condition": "on_complete",
        }

        plan["steps"].append({
            "type": "call_agent",
            "agent": "conductor-thinker",
            "role": "Thinker",
            "budget_tokens": 512,
        })

        plan["steps"].append({
            "type": cfg["type"],
            "agent": cfg["model"],
            "role": "Worker",
            "budget_tokens": 2048,
            "template": cfg["template"],
        })

        for vtype in cfg["verification"]:
            plan["verification_gates"].append({
                "type": vtype,
                "weight": cfg["verifier_weight"],
            })

        return plan
