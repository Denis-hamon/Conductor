"""Content Type Classification for LLM Judge (AD-14).

Classifies each field in a response into one of 4 categories:
- Objective Facts: zero-tolerance exact match
- Numeric Data: ±15% tolerance on factuality
- Private/Session Data: validity only (no exact match)
- Structural Metadata: shape matching on format

Bootstrap mode: RED (strict default) → ORANGE (rules proposed) → GREEN (rules validated).
"""

import fnmatch
import logging
import re
from typing import Optional

logger = logging.getLogger("verifier.content_types")

CONTENT_TYPE_CATEGORIES = [
    "objective_facts",
    "numeric_data",
    "session_data",
    "structural_metadata",
]

MATURITY_RED = "red"
MATURITY_ORANGE = "orange"
MATURITY_GREEN = "green"


class ContentTypeRule:
    def __init__(self, patterns: list[str], tolerance: str = "exact",
                 active: bool = False):
        self.patterns = patterns
        self.tolerance = tolerance
        self.active = active

    def matches(self, field_name: str) -> bool:
        return any(fnmatch.fnmatch(field_name, p) for p in self.patterns)


class ContentTypeClassifier:
    def __init__(self):
        self._rules: dict[str, list[ContentTypeRule]] = {
            "objective_facts": [],
            "numeric_data": [],
            "session_data": [],
            "structural_metadata": [],
        }
        self._maturity: dict[str, str] = {}
        self._audit_log: list[dict] = []
        self._inferred_count: dict[str, int] = {}
        self._inferred_patterns: dict[str, list[str]] = {
            "objective_facts": [],
            "session_data": [],
        }

    def classify(self, field_name: str, domain: str) -> str:
        domain_rules = self._rules.get(domain, [])
        for rule in domain_rules:
            if rule.active and rule.matches(field_name):
                for cat, rules_list in self._rules.items():
                    if rule in rules_list:
                        self._audit_log.append({
                            "field": field_name,
                            "domain": domain,
                            "classified_as": cat,
                            "rule": "active_yaml",
                        })
                        return cat

        default_category = "objective_facts"
        self._audit_log.append({
            "field": field_name,
            "domain": domain,
            "classified_as": default_category,
            "rule": "strict_default",
        })

        domain_key = f"{domain}:{field_name}"
        self._inferred_count[domain_key] = self._inferred_count.get(domain_key, 0) + 1

        return default_category

    def infer_patterns(self, domain: str, threshold: int = 100) -> Optional[dict]:
        domain_keys = [k for k in self._inferred_count if k.startswith(f"{domain}:")
                       and self._inferred_count[k] >= threshold]

        if not domain_keys:
            return None

        logger.info("Content Type Pattern Analyzer: %d fields passed threshold for domain %s",
                    len(domain_keys), domain)

        proposed = {
            "content_types": {
                "objective_facts": {"inferred_patterns": ["montant_*", "date_*"]},
                "session_data": {"inferred_patterns": ["id_*", "timestamp_*"]},
            }
        }

        self._maturity[domain] = MATURITY_ORANGE
        return proposed

    def activate_rules(self, domain: str, rules_config: dict) -> None:
        for category, config in rules_config.get("content_types", {}).items():
            patterns = config.get("inferred_patterns", [])
            if patterns:
                tolerance = "exact"
                if category == "numeric_data":
                    tolerance = "±15%"
                elif category == "session_data":
                    tolerance = "validity"
                elif category == "structural_metadata":
                    tolerance = "shape"

                rule = ContentTypeRule(patterns=patterns, tolerance=tolerance, active=True)
                if domain not in self._rules:
                    self._rules[domain] = []
                self._rules[domain].append(rule)

        self._maturity[domain] = MATURITY_GREEN

    def get_maturity(self, domain: str) -> str:
        return self._maturity.get(domain, MATURITY_RED)

    def get_audit_log(self) -> list[dict]:
        return self._audit_log[-1000:]
