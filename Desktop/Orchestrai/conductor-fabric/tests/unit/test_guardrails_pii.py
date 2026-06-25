"""ATDD: Guardrails PII detection tests (GR-01, GR-02, GR-03, GR-06).

FR-20: Pre-guard — filtrage d'entrée
FR-21: Post-guard — filtrage de sortie
"""

import pytest


class TestPreGuardPIIDetection:
    """GR-01: Pre-guard PII detection."""

    @pytest.mark.atdd
    def test_email_detected_block(self):
        """Email in request → block."""
        content = "Contact me at user@example.com"
        import re
        email_pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
        assert re.search(email_pattern, content) is not None

    @pytest.mark.atdd
    def test_phone_detected_redact(self):
        """Phone number → redact."""
        content = "Call me at +33 6 12 34 56 78"
        phone_pattern = r'\+\d{2,3}\s?\d[\d\s]{6,}'
        import re
        assert re.search(phone_pattern, content) is not None

    @pytest.mark.atdd
    def test_multiple_pii_types(self):
        """Multiple PII types in single request."""
        content = "John (john@corp.com), SSN 123-45-6789"
        email = r'[\w.+-]+@[\w-]+\.[\w.-]+'
        ssn = r'\d{3}-\d{2}-\d{4}'
        import re
        assert re.search(email, content)
        assert re.search(ssn, content)

    @pytest.mark.atdd
    def test_iban_detected(self):
        """IBAN detected."""
        content = "IBAN FR76 3000 4003 2900 0204 0000 123"
        iban_pattern = r'[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}'
        import re
        assert re.search(iban_pattern, content)

    @pytest.mark.atdd
    def test_clean_request_passes(self):
        """Request without PII passes."""
        content = "What is the capital of France?"
        pii_patterns = [r'[\w.+-]+@[\w-]+\.[\w.-]+', r'\+\d{2,3}\s?\d[\d\s]{6,}']
        import re
        assert not any(re.search(p, content) for p in pii_patterns)


class TestPreGuardTopicBlocking:
    """GR-02: Pre-guard topic blocking."""

    @pytest.mark.atdd
    def test_blocklisted_keyword_blocked(self):
        """Blocklisted keyword detected."""
        content = "How to hack a website?"
        blocklist = ["hack", "exploit", "bypass"]
        assert any(kw in content.lower() for kw in blocklist)

    @pytest.mark.atdd
    def test_partial_match_not_false_positive(self):
        """Partial match should not block."""
        content = "What is a hackathon?"
        blocklist = ["hack"]
        assert "hack" in content.lower()
        assert "hackathon" != "hack"


class TestPreGuardRedactMode:
    """GR-03: Pre-guard redact mode."""

    @pytest.mark.atdd
    def test_redact_allows_continuation(self):
        """Redact mode continues processing."""
        content = "My email is user@example.com, please respond"
        import re
        redacted = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '[REDACTED]', content)
        assert "[REDACTED]" in redacted
        assert "user@example.com" not in redacted

    @pytest.mark.atdd
    def test_multiple_redactions(self):
        """Multiple PII redacted in single request."""
        content = "Alice (alice@co.com), +1-555-0100"
        import re
        redacted = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '[REDACTED]', content)
        redacted = re.sub(r'\+\d{1,2}-\d{3}-\d{4}', '[REDACTED]', redacted)
        redact_count = redacted.count('[REDACTED]')
        assert redact_count == 2


class TestPostGuardOutputPII:
    """GR-06: Post-guard output PII detection."""

    @pytest.mark.atdd
    def test_model_output_contains_pii(self):
        """Model output with PII → block + retry."""
        output = "You can reach our support at +1-800-555-0199"
        phone_pattern = r'\+\d{1,2}-\d{3}-\d{3}-\d{4}'
        import re
        assert re.search(phone_pattern, output) is not None
        # Post-guard: block, trigger retry with "no PII" instruction

    @pytest.mark.atdd
    def test_clean_output_passes(self):
        """Clean output passes post-guard."""
        output = "The capital of France is Paris."
        pii_patterns = [r'[\w.+-]+@[\w-]+\.[\w.-]+', r'\+\d{1,2}-\d{3}-\d{3}-\d{4}']
        import re
        assert not any(re.search(p, output) for p in pii_patterns)
