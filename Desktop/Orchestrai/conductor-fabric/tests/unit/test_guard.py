"""Tests for Guard class — pre-guard and post-guard PII/topic detection."""

import pytest

from runtime.internal.guard.guardrails import Guard, GuardResult


class TestGuardCheckPII:
    def test_detects_email(self):
        g = Guard()
        found = g.check_pii("Contact me at user@example.com")
        assert "email" in found

    def test_detects_phone(self):
        g = Guard()
        found = g.check_pii("Call +33 6 12 34 56 78")
        assert "phone" in found

    def test_detects_ssn(self):
        g = Guard()
        found = g.check_pii("SSN 123-45-6789")
        assert "ssn" in found

    def test_detects_multiple_pii(self):
        g = Guard()
        found = g.check_pii("john@corp.com and +1-555-0100")
        assert "email" in found
        assert "phone" in found

    def test_clean_text_returns_empty(self):
        g = Guard()
        found = g.check_pii("What is the capital of France?")
        assert found == []


class TestGuardCheckTopics:
    def test_detects_blocked_topic(self):
        g = Guard()
        topic = g.check_topics("How to hack a website?")
        assert topic == "hack"

    def test_detects_exploit_keyword(self):
        g = Guard()
        topic = g.check_topics("bypass the firewall")
        assert topic == "bypass"

    def test_clean_text_returns_none(self):
        g = Guard()
        topic = g.check_topics("What is the weather?")
        assert topic is None

    def test_case_insensitive(self):
        g = Guard()
        topic = g.check_topics("How to HACK a server")
        assert topic == "hack"

    def test_partial_match_still_detects(self):
        g = Guard()
        topic = g.check_topics("hackathon project")
        assert topic == "hack"


class TestGuardPreGuard:
    def test_block_mode_blocks_topic(self):
        g = Guard(mode="block")
        result = g.pre_guard("How to hack a system")
        assert result.blocked is True
        assert result.action == "block"
        assert result.topic_found == "hack"

    def test_block_mode_blocks_pii(self):
        g = Guard(mode="block")
        result = g.pre_guard("My email is user@example.com")
        assert result.blocked is True
        assert result.action == "block"
        assert "email" in result.pii_found

    def test_redact_mode_allows_pii(self):
        g = Guard(mode="redact")
        result = g.pre_guard("My email is user@example.com")
        assert result.blocked is False
        assert result.action == "redact"
        assert "email" in result.pii_found

    def test_clean_request_passes(self):
        g = Guard()
        result = g.pre_guard("What is the capital of France?")
        assert result.blocked is False
        assert result.action == "pass"

    def test_topic_takes_precedence_over_pii(self):
        g = Guard(mode="redact")
        result = g.pre_guard("How to hack a system, email: user@example.com")
        assert result.blocked is True
        assert result.action == "block"
        assert result.topic_found == "hack"

    def test_guard_result_to_dict(self):
        result = GuardResult(blocked=True, reason="PII detected", pii_found=["email"], action="block")
        d = result.to_dict()
        assert d["blocked"] is True
        assert d["reason"] == "PII detected"
        assert d["pii_found"] == ["email"]
        assert d["action"] == "block"


class TestGuardPostGuard:
    def test_blocks_output_pii(self):
        g = Guard()
        result = g.post_guard("You can reach us at +1-800-555-0199")
        assert result.blocked is True
        assert result.action == "block"

    def test_clean_output_passes(self):
        g = Guard()
        result = g.post_guard("The capital of France is Paris.")
        assert result.blocked is False
        assert result.action == "pass"

    def test_default_mode_is_block(self):
        g = Guard()
        assert g.mode == "block"


class TestGuardDefaultMode:
    def test_default_mode_is_block(self):
        g = Guard()
        assert g.mode == "block"

    def test_custom_mode(self):
        g = Guard(mode="redact")
        assert g.mode == "redact"
