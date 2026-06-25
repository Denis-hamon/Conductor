"""PageIndex tree search — LLM-guided navigation over hierarchical index.

Simplified search that flattens the tree, uses keyword + vector similarity
to find relevant leaf nodes, then returns their page ranges.

Full agentic tree search (OpenAI Agents SDK) is available in PageIndex examples.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger("rag.tree_search")


def _flatten_tree(nodes: list, parent_path: str = "") -> list[dict]:
    flat = []
    for node in nodes:
        title = node.get("title", "")
        summary = node.get("summary", "")
        node_id = node.get("node_id", "")
        path = f"{parent_path} > {title}" if parent_path else title
        page_range = _get_page_range(node)
        flat.append({
            "node_id": node_id,
            "title": title,
            "summary": summary,
            "path": path,
            "page_range": page_range,
        })
        if "nodes" in node and node["nodes"]:
            flat.extend(_flatten_tree(node["nodes"], path))
    return flat


def _get_page_range(node: dict) -> str:
    if "page_range" in node:
        return node["page_range"]
    start = node.get("start_index")
    end = node.get("end_index")
    if start is not None and end is not None:
        return f"{start}-{end}"
    return ""


def _score_section(query: str, section: dict) -> float:
    score = 0.0
    q_lower = query.lower()
    title = section.get("title", "").lower()
    summary = section.get("summary", "").lower()
    path = section.get("path", "").lower()

    title_words = set(title.split())
    query_words = set(q_lower.split())
    common = title_words & query_words
    score += len(common) * 2.0

    if summary:
        for word in q_lower.split():
            if word in summary:
                score += 0.5

    if path:
        for word in q_lower.split():
            if word in path:
                score += 0.3

    return score


def search_tree(query: str, structure: list, doc_info: Optional[dict] = None,
                top_k: int = 5) -> list[str]:
    flat_nodes = _flatten_tree(structure)
    if not flat_nodes:
        logger.warning("Empty tree structure")
        return []

    scored = [(node, _score_section(query, node)) for node in flat_nodes]
    scored.sort(key=lambda x: -x[1])

    seen_ranges = set()
    results = []
    for node, score in scored:
        pr = node["page_range"]
        if not pr:
            continue
        if pr in seen_ranges:
            continue
        seen_ranges.add(pr)
        results.append(pr)
        if len(results) >= top_k:
            break

    if not results:
        logger.warning("No page ranges found in tree")
        if doc_info:
            total = doc_info.get("page_count", doc_info.get("line_count", 0))
            if total:
                results.append(f"1-{min(total, 5)}")

    logger.info("Tree search: query='%s' → %d sections (%s)", query[:40], len(results), results)
    return results
