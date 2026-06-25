"""Pre-guard and Post-guard for PII detection and topic blocking."""

import logging
import re
from typing import Optional

logger = logging.getLogger("guard")

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}")
SSN_PATTERN = re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b")

BLOCKED_TOPICS = [
    "hack", "crack", "exploit", "bypass", "malware", "ransomware",
    "phishing", "social engineering", "unauthorized access",
]

PII_PATTERNS = [
    ("email", EMAIL_PATTERN),
    ("phone", PHONE_PATTERN),
    ("ssn", SSN_PATTERN),
]


class GuardResult:
    def __init__(self, blocked: bool, reason: str = "", pii_found: list[str] = None,
                 topic_found: str = "", action: str = "pass"):
        self.blocked = blocked
        self.reason = reason
        self.pii_found = pii_found or []
        self.topic_found = topic_found
        self.action = action

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "pii_found": self.pii_found,
            "topic_found": self.topic_found,
            "action": self.action,
        }


class Guard:
    def __init__(self, mode: str = "block"):
        self.mode = mode

    def check_pii(self, text: str) -> list[str]:
        found = []
        for name, pattern in PII_PATTERNS:
            if pattern.search(text):
                found.append(name)
        return found

    def check_topics(self, text: str) -> Optional[str]:
        lower = text.lower()
        for topic in BLOCKED_TOPICS:
            if topic in lower:
                return topic
        return None

    def pre_guard(self, text: str) -> GuardResult:
        pii = self.check_pii(text)
        topic = self.check_topics(text)

        if topic:
            return GuardResult(blocked=True, reason="Topic blocked",
                               topic_found=topic, action="block")

        if pii:
            if self.mode == "block":
                return GuardResult(blocked=True, reason="PII detected",
                                   pii_found=pii, action="block")
            return GuardResult(blocked=False, reason="PII redacted",
                               pii_found=pii, action="redact")

        return GuardResult(blocked=False, action="pass")

    def post_guard(self, text: str) -> GuardResult:
        pii = self.check_pii(text)

        if pii:
            return GuardResult(blocked=True, reason="PII detected in output",
                               pii_found=pii, action="block")

        return GuardResult(blocked=False, action="pass")
