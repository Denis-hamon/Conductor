"""PromptAssembler Stage 1 — Template resolution with inheritance and overlay (AD-12).

Resolves {domain}/{env} templates by traversing the inheritance chain.
Supports `extends:` declarations for partial overrides.
"""

import hashlib
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger("assembler.stage1")

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "prompts")


class ResolvedTemplate:
    def __init__(self, content: str, prompt_id: str, chain: list[str]):
        self.content = content
        self.prompt_id = prompt_id
        self.chain = chain

    def to_dict(self) -> dict:
        return {"content": self.content, "prompt_id": self.prompt_id, "chain": self.chain}


class TemplateResolver:
    def __init__(self, prompts_root: str = PROMPTS_DIR):
        self.prompts_root = prompts_root
        self._cache: dict[str, ResolvedTemplate] = {}

    def resolve(self, domain: str, env: str) -> ResolvedTemplate:
        cache_key = f"{domain}:{env}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        chain = self._build_chain(domain)
        resolved = self._merge_chain(chain, env)
        prompt_id = self._compute_id(resolved, chain)

        result = ResolvedTemplate(content=resolved, prompt_id=prompt_id, chain=chain)
        self._cache[cache_key] = result
        return result

    def invalidate_cache(self, domain: Optional[str] = None, env: Optional[str] = None) -> None:
        if domain and env:
            self._cache.pop(f"{domain}:{env}", None)
        elif domain:
            self._cache = {k: v for k, v in self._cache.items() if not k.startswith(f"{domain}:")}
        else:
            self._cache.clear()

    def _build_chain(self, domain: str) -> list[str]:
        chain = ["shared"]
        parts = domain.split("_")
        if len(parts) > 1:
            for i in range(1, len(parts)):
                chain.append("_".join(parts[:i]))
        if domain != "shared":
            chain.append(domain)
        return chain

    def _merge_chain(self, chain: list[str], env: str) -> str:
        base_content = self._load_template("shared", env)
        if not base_content:
            return ""

        sections = self._parse_sections(base_content)

        for domain in chain[1:]:
            domain_content = self._load_template(domain, env)
            if domain_content is None:
                continue

            if self._has_extends(domain_content, env):
                domain_sections = self._parse_sections(domain_content)
                sections.update(domain_sections)
            else:
                sections = self._parse_sections(domain_content)

        return self._rebuild(sections)

    def _load_template(self, domain: str, env: str) -> Optional[str]:
        for name in (domain, env):
            if ".." in name or name.startswith("/"):
                logger.warning("blocked path traversal: domain=%s env=%s", domain, env)
                return None

        root = os.path.realpath(self.prompts_root)
        path = os.path.realpath(os.path.join(root, domain, f"{env}.md"))
        if not path.startswith(root):
            logger.warning("blocked path escape: %s", path)
            return None
        if os.path.exists(path):
            with open(path) as f:
                return f.read()

        alt_path = os.path.realpath(os.path.join(root, "shared", "world-models", f"{env}.md"))
        if domain == "shared" and alt_path.startswith(root) and os.path.exists(alt_path):
            with open(alt_path) as f:
                return f.read()

        return None

    def _has_extends(self, content: str, env: str) -> bool:
        return f"extends: shared/world-models/{env}" in content or "extends:" in content

    def _parse_sections(self, content: str) -> dict[str, str]:
        sections = {}
        lines = content.split("\n")
        current_section = "__header__"
        current_lines = []

        for line in lines:
            if line.startswith("### ") or line.startswith("## ") or line.startswith("# "):
                if current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = line.strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        return sections

    def _rebuild(self, sections: dict[str, str]) -> str:
        return "\n\n".join(sections.values())

    def _compute_id(self, content: str, chain: list[str]) -> str:
        raw = f"{content}|{'->'.join(chain)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
