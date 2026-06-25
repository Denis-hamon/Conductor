"""AgentWorldBench — judge output parser (adapted from QwenLM eval)."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

try:
    from .task_configs import SCORE_DIMENSIONS, TASK_CONFIGS
    from .output_parser import remove_thinking_tags
except ImportError:
    from task_configs import SCORE_DIMENSIONS, TASK_CONFIGS
    from output_parser import remove_thinking_tags

logger = logging.getLogger("agentworld.judge_parser")
_PROMPTS_DIR = Path(__file__).absolute().parent / "prompts"


def parse_judge_output(raw_output: str, response_tag: str = "final_evaluation") -> dict:
    try:
        cleaned = (raw_output or "").strip()
        if not cleaned:
            return _error("Empty output", raw_output)
        cleaned = remove_thinking_tags(cleaned, response_tag)
        json_content = _extract_tagged(cleaned, response_tag)
        text = json_content if json_content else cleaned
        json_str = (
            _extract_from_markdown(text)
            or _extract_best_json(text)
            or _extract_last_json(text)
        )
        if not json_str and json_content:
            json_str = _extract_best_json(json_content) or _extract_last_json(json_content)
        if not json_str:
            return _error("No JSON found", raw_output)
        json_str = _repair_json(json_str)
        result = json.loads(json_str)
        scores = _extract_scores(result)
        if not scores:
            return _error(f"Invalid scores. Keys: {list(result.keys())}", raw_output)
        valid = [v for v in scores.values() if v > 0]
        return {
            "strengths": _to_list(result.get("strengths", [])),
            "weaknesses": _to_list(result.get("weaknesses", [])),
            "scores": scores,
            "total_score": sum(valid) / len(valid) if valid else 0,
            "success": True,
            "judge_raw_output": raw_output,
        }
    except json.JSONDecodeError as e:
        return _error(f"JSON error: {e}", raw_output)
    except Exception as e:
        return _error(f"Error: {e}", raw_output)


def load_judge_system_prompts() -> dict:
    prompts = {}
    for subtask, config in TASK_CONFIGS.items():
        path = config.get("judge_system_prompt_path")
        if not path:
            prompts[subtask] = ""
            continue
        full = _PROMPTS_DIR / path.replace("prompts/", "")
        if full.exists():
            prompts[subtask] = full.read_text(encoding="utf-8")
        else:
            prompts[subtask] = ""
    return prompts


def _extract_tagged(text: str, tag: str) -> Optional[str]:
    if not tag:
        return None
    starts = list(re.finditer(rf"<{re.escape(tag)}>", text, re.IGNORECASE))
    if not starts:
        return None
    start = starts[-1].end()
    close = re.search(rf"</{re.escape(tag)}>", text[start:], re.IGNORECASE)
    if close:
        return text[start:start + close.start()].strip()
    return text[start:].strip()


def _extract_from_markdown(text: str) -> Optional[str]:
    matches = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text))
    if not matches:
        return None
    for m in reversed(matches):
        c = m.group(1).strip()
        if c.startswith("{") and c.endswith("}") and '"scores"' in c:
            return c
    for m in reversed(matches):
        c = m.group(1).strip()
        if c.startswith("{") and c.endswith("}"):
            return c
    return None


def _extract_best_json(text: str) -> Optional[str]:
    objs = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        obj = _match_braces(text, start)
        if obj:
            objs.append(obj)
            pos = start + len(obj)
        else:
            pos = start + 1
    for obj in reversed(objs):
        if '"scores"' in obj:
            return obj
    return objs[-1] if objs else None


def _extract_last_json(text: str) -> Optional[str]:
    end = text.rfind("}")
    if end == -1:
        return None
    depth = in_str = i = 0
    in_str = False
    depth = 0
    i = end
    while i >= 0:
        c = text[i]
        if c == '"' and (i == 0 or text[i - 1] != "\\"):
            in_str = not in_str
        if not in_str:
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    return text[i:end + 1]
        i -= 1
    return None


def _match_braces(text: str, start: int) -> Optional[str]:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_str:
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair_json(s: str) -> str:
    s = re.sub(r',(\s*[}\]])', r"\1", s)
    s = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', s)
    s = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', s)
    return s


def _extract_scores(result: dict) -> Optional[dict]:
    if "scores" not in result or not isinstance(result["scores"], dict):
        return None
    scores = {}
    for dim in SCORE_DIMENSIONS:
        v = result["scores"].get(dim, 0)
        if isinstance(v, int):
            scores[dim] = v
        elif isinstance(v, float):
            scores[dim] = int(round(v))
        elif isinstance(v, str):
            try:
                scores[dim] = int(v.split("/")[0].strip())
            except (ValueError, TypeError):
                scores[dim] = 0
        else:
            scores[dim] = 0
        if scores[dim] > 0:
            scores[dim] = max(1, min(5, scores[dim]))
    return scores


def _to_list(value: Any) -> list:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [value] if value else []
    return []


def _error(msg: str, raw: str = "") -> dict:
    return {
        "error_message": msg,
        "strengths": [],
        "weaknesses": [],
        "scores": {d: 0 for d in SCORE_DIMENSIONS},
        "total_score": 0.0,
        "success": False,
        "judge_raw_output": raw,
    }
