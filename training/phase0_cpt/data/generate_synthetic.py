"""Generate synthetic workflow traces for CPT pipeline testing.

Produces Parquet files matching the TraceStore.build_trace() format,
with realistic-but-synthetic requests, plans, steps, and rewards.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("phase0_cpt.synthetic")

DOMAINS = [
    "code",
    "rag",
    "reason",
    "general",
    "code",
    "rag",
    "reason",
    "mcp",
]

DIFFICULTIES = ["easy", "medium", "hard"]

STEP_CONFIGS: dict[str, list[dict]] = {
    "code": [
        {"type": "call_agent", "agent": "conductor-thinker", "role": "Thinker", "budget_tokens": 512},
        {"type": "call_agent", "agent": "qwen-coder", "role": "Worker", "budget_tokens": 2048},
    ],
    "rag": [
        {"type": "call_agent", "agent": "conductor-thinker", "role": "Thinker", "budget_tokens": 512},
        {"type": "call_agent", "agent": "qwen-reasoner", "role": "Worker", "budget_tokens": 2048},
    ],
    "reason": [
        {"type": "call_agent", "agent": "conductor-thinker", "role": "Thinker", "budget_tokens": 1024},
        {"type": "call_agent", "agent": "qwen-reasoner", "role": "Worker", "budget_tokens": 4096},
    ],
    "general": [
        {"type": "call_agent", "agent": "conductor-thinker", "role": "Thinker", "budget_tokens": 256},
        {"type": "call_agent", "agent": "gemma-3-small", "role": "Worker", "budget_tokens": 1024},
    ],
    "mcp": [
        {"type": "call_agent", "agent": "conductor-thinker", "role": "Thinker", "budget_tokens": 512},
        {"type": "call_agent", "agent": "qwen-reasoner", "role": "Worker", "budget_tokens": 2048},
    ],
}

VERIFICATION_MAP: dict[str, list[dict]] = {
    "code": [{"type": "sandbox_code", "weight": 0.8}],
    "rag": [{"type": "rag_citation", "weight": 0.7}],
    "reason": [{"type": "llm_judge", "weight": 0.6}],
    "general": [{"type": "llm_judge", "weight": 0.3}],
    "mcp": [{"type": "simulated", "weight": 0.5}],
}

REQUEST_TEMPLATES: dict[str, list[str]] = {
    "code": [
        "Write a Python function to sort a list of dictionaries by a key",
        "Implement a binary search tree in JavaScript",
        "Write a SQL query to find duplicate email addresses",
        "Create a bash script to backup PostgreSQL databases",
        "Implement a rate limiter using a token bucket algorithm",
    ],
    "rag": [
        "According to the contract, what is the termination clause?",
        "Find the section about data retention in the privacy policy",
        "Summarize the key findings from the research paper",
        "Extract all clauses related to intellectual property",
        "What does the compliance document say about GDPR?",
    ],
    "reason": [
        "Solve the equation 3x^2 + 5x - 2 = 0",
        "Compare the efficiency of quicksort and mergesort for large datasets",
        "Explain why the sky appears blue during the day",
        "Analyze the time complexity of this algorithm: O(n log n)",
        "Evaluate the ethical implications of facial recognition technology",
    ],
    "general": [
        "What is the capital of France?",
        "Explain how photosynthesis works",
        "Give me tips for learning a new language",
        "What are the benefits of regular exercise?",
        "Write a short poem about autumn",
    ],
    "mcp": [
        "Call the weather API to get the forecast for Paris",
        "Use the database tool to query recent orders",
        "Execute the search function to find relevant documents",
        "Call the Slack API to send a notification",
    ],
}


def _generate_plan(domain: str) -> dict:
    steps = STEP_CONFIGS.get(domain, STEP_CONFIGS["general"])
    verification = VERIFICATION_MAP.get(domain, VERIFICATION_MAP["general"])
    return {
        "domain": domain,
        "steps": [dict(s) for s in steps],
        "verification_gates": [dict(v) for v in verification],
        "stop_condition": "on_complete",
    }


def _generate_steps(domain: str, difficulty: str) -> list[dict]:
    base = STEP_CONFIGS.get(domain, STEP_CONFIGS["general"])
    if difficulty == "hard":
        extra = {"type": "verify", "agent": "qwen-reasoner", "role": "Verifier", "budget_tokens": 1024}
        return base + [extra]
    return base


def _generate_rubric(difficulty: str) -> dict:
    reward_map = {"easy": 1.0, "medium": 0.8, "hard": 0.6}
    return {
        "reward": reward_map.get(difficulty, 0.5),
        "safety": 1.0,
        "factuality": 0.9,
        "fidelity": 0.85,
    }


def generate_trace(
    domain: str,
    difficulty: str,
    idx: int,
    base_ts: int,
) -> dict:
    request_text = REQUEST_TEMPLATES[domain][idx % len(REQUEST_TEMPLATES[domain])]
    request = {
        "model": "conductor-fabric",
        "messages": [{"role": "user", "content": request_text}],
    }
    plan = _generate_plan(domain)
    steps = _generate_steps(domain, difficulty)
    rubric = _generate_rubric(difficulty)

    return {
        "trace_id": f"syn_tr_{idx:06d}",
        "workflow_id": f"syn_wf_{idx:06d}",
        "request": request,
        "workflow_plan": plan,
        "steps": [
            {
                "step_id": f"s{i}-{idx}",
                "type": s["type"],
                "agent": s.get("agent", ""),
                "output": f"Output from {s.get('role', 'worker')} for {request_text[:30]}...",
                "duration_ms": 500 + (i * 200),
            }
            for i, s in enumerate(steps)
        ],
        "rubric": rubric,
        "reward": rubric["reward"],
        "stop_reason": "on_complete",
        "metadata": {
            "domain": domain,
            "difficulty": difficulty,
            "source": "synthetic",
            "tenant": "default-tenant",
        },
        "otel_compatible": True,
        "stored_at": base_ts + idx * 60,
        "retention_days": 90,
    }


def generate_dataset(
    output_dir: str,
    num_traces: int = 10000,
    seed: int = 42,
) -> str:
    import random
    random.seed(seed)

    base_ts = int((datetime.now() - timedelta(days=7)).timestamp())
    traces: list[dict[str, Any]] = []

    for i in range(num_traces):
        domain = random.choice(DOMAINS)
        difficulty = random.choice(DIFFICULTIES)
        trace = generate_trace(domain, difficulty, i, base_ts)
        traces.append(trace)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(traces)
    out_path = str(out_dir / f"synthetic_traces_{num_traces}.parquet")
    pq.write_table(table, out_path)

    meta = {
        "num_traces": num_traces,
        "seed": seed,
        "output": out_path,
        "domains": list(set(DOMAINS)),
    }
    meta_path = out_dir / "generation_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Generated %d synthetic traces → %s", num_traces, out_path)
    return out_path
