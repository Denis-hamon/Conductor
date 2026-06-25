"""Format workflow traces into CPT training sequences.

Converts Parquet traces into (query → plan → steps → reward) sequences
for continued pre-training of small language models.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

logger = logging.getLogger("phase0_cpt.format")


@dataclass
class TrainingExample:
    query: str
    plan: dict[str, Any]
    steps: list[dict[str, Any]]
    reward: float
    domain: str
    difficulty: str
    sequence: str = ""


TEMPLATE = """### Request
{query}

### Plan
{plan_json}

### Steps
{steps_json}

### Reward
{reward}

### Domain
{domain}
"""


def load_traces(path: str) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def _try_parse_json(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val


def format_example(trace: dict[str, Any]) -> Optional[TrainingExample]:
    request = _try_parse_json(trace.get("request"))
    plan = _try_parse_json(trace.get("workflow_plan"))
    steps = _try_parse_json(trace.get("steps")) or trace.get("steps", [])
    metadata = _try_parse_json(trace.get("metadata", {}))
    rubric = _try_parse_json(trace.get("rubric", {}))

    if not request or not plan:
        return None

    query = ""
    if isinstance(request, dict):
        messages = request.get("messages", [])
        if messages:
            query = messages[-1].get("content", "")
    elif isinstance(request, str):
        query = request

    reward = rubric.get("reward", 0.0) if isinstance(rubric, dict) else 0.0

    return TrainingExample(
        query=query,
        plan=plan,
        steps=steps or [],
        reward=reward,
        domain=metadata.get("domain", "general"),
        difficulty=metadata.get("difficulty", "medium"),
        sequence=TEMPLATE.format(
            query=query,
            plan_json=_safe_json(plan),
            steps_json=_safe_json(steps or []),
            reward=reward,
            domain=metadata.get("domain", "general"),
        ),
    )


def format_dataset(
    input_path: str,
    output_path: str,
    min_reward: float = -1.0,
    max_examples: Optional[int] = None,
) -> int:
    traces = load_traces(input_path)
    logger.info("Loaded %d traces from %s", len(traces), input_path)

    examples: list[TrainingExample] = []
    for trace in traces:
        example = format_example(trace)
        if example is None:
            continue
        if example.reward < min_reward:
            continue
        examples.append(example)

    if max_examples:
        examples = examples[:max_examples]

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = [e.sequence for e in examples]
    out_file = out_dir / "train_sequences.jsonl"
    with open(out_file, "w") as f:
        for seq in sequences:
            f.write(json.dumps({"text": seq}) + "\n")

    meta = {
        "input": str(input_path),
        "output": str(out_file),
        "total_traces": len(traces),
        "formatted_examples": len(examples),
        "min_reward": min_reward,
    }
    meta_file = out_dir / "format_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Formatted %d/%d examples → %s",
        len(examples),
        len(traces),
        out_file,
    )
    return len(examples)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
