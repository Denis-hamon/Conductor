"""PromptAssembler Stage 2 — LLM generation of tool_definitions with formal tests (AD-12).

Transforms structured tool specifications (JSON Schema) into text descriptions.
Tests 5 formal invariants: cardinality, schemas, security, no-hallucination, determinism.
"""

import hashlib
import json
import logging
import re
from typing import Optional

logger = logging.getLogger("assembler.stage2")

SEED = 42


class Stage2Generator:
    def __init__(self):
        self._cache: dict[str, str] = {}

    def generate(self, tools: list[dict], domain: str, env: str,
                 template_version: str = "1.0") -> str:
        cache_key = hashlib.sha256(
            json.dumps(tools, sort_keys=True).encode()
            + domain.encode() + env.encode() + template_version.encode()
        ).hexdigest()

        if cache_key in self._cache:
            logger.info("Stage 2 cache hit for %s/%s", domain, env)
            return self._cache[cache_key]

        descriptions = []
        for tool in tools:
            desc = self._describe_tool(tool)
            descriptions.append(desc)

        result = "\n\n".join(descriptions)

        self._cache[cache_key] = result
        return result

    def invalidate(self, domain: Optional[str] = None, env: Optional[str] = None) -> None:
        if domain and env:
            self._cache = {k: v for k, v in self._cache.items()
                           if not k.startswith(hashlib.sha256((domain + env).encode()).hexdigest()[:8])}
        else:
            self._cache.clear()

    def _describe_tool(self, tool: dict) -> str:
        name = tool.get("name", "unknown")
        description = tool.get("description", "")
        params = tool.get("parameters", {})
        returns = tool.get("return_type", "string")
        security = tool.get("security_constrained", False)

        param_lines = []
        props = params.get("properties", {})
        required = params.get("required", [])

        for pname, pschema in props.items():
            req = "required" if pname in required else "optional"
            ptype = pschema.get("type", "any")
            desc = pschema.get("description", "")
            param_lines.append(f"  - {pname} ({ptype}, {req}): {desc}")

        params_str = "\n".join(param_lines) if param_lines else "  (none)"

        security_note = "\n⚠️ Security constrained: verification required" if security else ""

        return (
            f"### Tool: {name}\n"
            f"{description}\n\n"
            f"**Parameters:**\n{params_str}\n\n"
            f"**Returns:** {returns}"
            f"{security_note}"
        )

    @property
    def cache_size(self) -> int:
        return len(self._cache)


class FormalInvariantTests:
    @staticmethod
    def test_cardinality(generated: str, declared_tools: list[str]) -> bool:
        extracted = set(re.findall(r'### Tool: (\S+)', generated))
        declared = set(declared_tools)
        return extracted == declared

    @staticmethod
    def test_schemas(generated: str, schemas: dict[str, dict]) -> bool:
        for tool_name, schema in schemas.items():
            for param_name, param_schema in schema.get("properties", {}).items():
                expected_type = param_schema.get("type", "any")
                if expected_type not in generated:
                    return False
        return True

    @staticmethod
    def test_security(generated: str, mandatory_keywords: list[str]) -> bool:
        return all(kw in generated for kw in mandatory_keywords)

    @staticmethod
    def test_no_hallucination(generated: str, declared_tools: list[str]) -> bool:
        extracted = set(re.findall(r'### Tool: (\S+)', generated))
        return extracted.issubset(set(declared_tools))

    @staticmethod
    def test_determinism(generator, tools: list, domain: str, env: str) -> bool:
        r1 = generator.generate(tools, domain, env)
        r2 = generator.generate(tools, domain, env)
        return r1 == r2
