"""AgentWorldBench — output parsing utilities (adapted from QwenLM eval)."""

import re
from pathlib import Path

try:
    from .task_configs import TASK_CONFIGS
except ImportError:
    from task_configs import TASK_CONFIGS

_UTILS_DIR = Path(__file__).absolute().parent


_RE_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_RE_THINK_CLOSE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


def remove_thinking_tags(text: str, response_tag: str) -> str:
    if not text:
        return text
    tags = []
    for m in _RE_THINK_OPEN.finditer(text):
        tags.append((m.start(), "open", m.end()))
    for m in _RE_THINK_CLOSE.finditer(text):
        tags.append((m.start(), "close", m.end()))
    if not tags:
        return text
    tags.sort(key=lambda x: x[0])
    ranges_to_remove = []
    used = set()
    for i, (pos, kind, end) in enumerate(tags):
        if i in used:
            continue
        if kind == "open":
            for j in range(i + 1, len(tags)):
                if j in used:
                    continue
                if tags[j][1] == "close":
                    ranges_to_remove.append((pos, tags[j][2]))
                    used.add(i)
                    used.add(j)
                    break
    for i, (pos, kind, end) in enumerate(tags):
        if i in used:
            continue
        if kind == "close":
            ranges_to_remove.append((0, end))
            used.add(i)
    for i, (pos, kind, end) in enumerate(tags):
        if i in used:
            continue
        if kind == "open":
            stop = len(text)
            if response_tag:
                m = re.search(rf"<{re.escape(response_tag)}>", text[pos:], re.IGNORECASE)
                if m:
                    stop = pos + m.start()
            ranges_to_remove.append((pos, stop))
            used.add(i)
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
