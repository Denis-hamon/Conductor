"""Tests for SimVerif — simulation-based verification via AgentWorld."""

import pytest

from runtime.internal.verifier.simulation.simverif import SimVerif, SimVerifResult


@pytest.fixture
def simverif():
    return SimVerif(agentworld_url="http://localhost:1")


class TestSimVerifResult:
    def test_simulated_result_to_dict(self):
        result = SimVerifResult(simulated=True, score=1.0, confidence=0.85,
                                output="OK", latency_ms=150.0)
        d = result.to_dict()
        assert d["verification_type"] == "simulated"
        assert d["score"] == 1.0
        assert d["confidence"] == 0.85
        assert d["latency_ms"] == 150.0

    def test_fallback_result_to_dict(self):
        result = SimVerifResult(simulated=False, score=0.0, confidence=0.0,
                                fallback_reason="timeout", latency_ms=5000.0)
        d = result.to_dict()
        assert d["verification_type"] == "fallback"
        assert d["fallback_reason"] == "timeout"


class TestSimVerifConfidence:
    def test_estimate_confidence_default(self, simverif):
        confidence = simverif._estimate_confidence("some response")
        assert confidence == 0.85

    def test_confidence_threshold_is_0_7(self):
        from runtime.internal.verifier.simulation.simverif import CONFIDENCE_THRESHOLD
        assert CONFIDENCE_THRESHOLD == 0.7


class TestSimVerifSystemPrompts:
    def test_get_system_prompt_swe(self, simverif):
        prompt = simverif._get_system_prompt("swe")
        assert "software engineering" in prompt.lower()

    def test_get_system_prompt_mcp(self, simverif):
        prompt = simverif._get_system_prompt("mcp")
        assert "mcp tool" in prompt.lower()

    def test_get_system_prompt_terminal(self, simverif):
        prompt = simverif._get_system_prompt("terminal")
        assert "terminal" in prompt.lower()

    def test_get_system_prompt_unknown_env(self, simverif):
        prompt = simverif._get_system_prompt("unknown")
        assert "terminal" in prompt.lower()


@pytest.mark.asyncio
class TestSimVerifVerify:
    async def test_verify_code_returns_fallback_on_timeout(self, simverif):
        result = await simverif.verify_code("def add(a, b): return a + b", domain="code")
        assert result.simulated is False
        assert result.score == 0.0

    async def test_verify_mcp_returns_fallback_on_failure(self, simverif):
        result = await simverif.verify_mcp("search", {"q": "test"})
        assert result.simulated is False
        assert result.score == 0.0
