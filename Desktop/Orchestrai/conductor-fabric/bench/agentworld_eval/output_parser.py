"""AgentWorldBench — output parsing utilities (adapted from QwenLM eval)."""

import re
from pathlib import Path

from .task_configs import TASK_CONFIGS

_UTILS_DIR = Path(__file__).absolute().parent


def remove_thinking_tags(text: str, response_tag: str) -> str:
    if not text:
        return text
    tags = []
    for m in re.finditer(r"<think>", text, re.IGNORECASE):
        tags.append((m.start(), "open"))
    for m in re.finditer(r"</think>", text, re.IGNORECASE):
        tags.append((m.start(), "close"))
    if not tags:
        return text
    tags.sort(key=lambda x: x[0])
    ranges_to_remove = []
    used_indices = set()
    for i, (pos, tag_type) in enumerate(tags):
        if i in used_indices:
            continue
        if tag_type == "open":
            for j in range(i + 1, len(tags)):
                if j in used_indices:
                    continue
                if tags[j][1] == "close":
                    ranges_to_remove.append((pos, tags[j][0] + len("</think>")))
                    used_indices.add(i)
                    used_indices.add(j)
                    break
    for i, (pos, tag_type) in enumerate(tags):
        if i in used_indices:
            continue
        if tag_type == "close":
            ranges_to_remove.append((0, pos + len("</think>")))
            used_indices.add(i)
    for i, (pos, tag_type) in enumerate(tags):
        if i in used_indices:
            continue
        if tag_type == "open":
            end_pos = len(text)
            if response_tag:
                m = re.search(rf"<{re.escape(response_tag)}>", text[pos:], re.IGNORECASE)
                if m:
                    end_pos = pos + m.start()
            ranges_to_remove.append((pos, end_pos))
            used_indices.add(i)
    if not ranges_to_remove:
        return text
    ranges_to_remove.sort()
    merged = []
    for s, e in ranges_to_remove:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    parts, prev = [], 0
    for s, e in merged:
        if s > prev:
            parts.append(text[prev:s])
        prev = e
    if prev < len(text):
        parts.append(text[prev:])
    return "".join(parts).strip()


def parse_model_output(raw_output: str, response_tag: str) -> str:
    if not raw_output:
        return "No output"
    cleaned = remove_thinking_tags(raw_output, response_tag)
    pattern = rf"<{re.escape(response_tag)}>"
    matches = list(re.finditer(pattern, cleaned, re.IGNORECASE))
    if not matches:
        return cleaned.strip() or "No output"
    last = matches[-1]
    start = last.end()
    close = re.search(rf"</{re.escape(response_tag)}>", cleaned[start:], re.IGNORECASE)
    if close:
        return cleaned[start:start + close.start()].strip()
    return cleaned[start:].strip()


def clean_response_marker(text: str, subtask: str) -> str:
    marker = TASK_CONFIGS.get(subtask, {}).get("response_marker", "")
    if marker:
        return text.replace(marker, "").strip()
    return text.strip()
