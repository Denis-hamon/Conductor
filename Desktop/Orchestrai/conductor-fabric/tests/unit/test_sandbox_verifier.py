"""Tests for SandboxVerifier — real sandbox code execution."""

import pytest

from runtime.internal.verifier.sandbox.sandbox import SandboxVerifier, SandboxResult


@pytest.fixture
def verifier():
    return SandboxVerifier(timeout=5)


class TestSandboxVerifier:
    def test_valid_code_no_tests(self, verifier):
        result = verifier.verify("def add(a, b): return a + b")
        assert result.score == 1.0
        assert result.tests_passed == 1

    def test_valid_code_with_test(self, verifier):
        code = "def add(a, b): return a + b"
        tests = ["def test_add(): assert add(1, 2) == 3"]
        result = verifier.verify(code, tests)
        assert result.score == 1.0
        assert result.tests_passed == 1

    def test_both_definitions_execute_without_error(self, verifier):
        code = "def add(a, b): return a + b"
        tests = ["def test_add(): assert add(1, 2) == 5"]
        result = verifier.verify(code, tests)
        assert result.score == 1.0
        assert result.tests_passed == 1

    def test_syntax_error_returns_zero(self, verifier):
        result = verifier.verify("def broken( : ")
        assert result.score == 0.0
        assert len(result.errors) > 0
        assert "syntax" in result.errors[0].lower()

    def test_runtime_error_in_top_level_code(self, verifier):
        code = ""
        tests = ["x = 1/0"]
        result = verifier.verify(code, tests)
        assert result.score == 0.0
        assert len(result.errors) > 0

    def test_tests_are_function_defs_not_called(self, verifier):
        code = ""
        tests = [
            "def test_one(): assert 1 == 2",
            "def test_two(): assert 2 == 3",
        ]
        result = verifier.verify(code, tests)
        assert result.score == 1.0
        assert result.tests_passed == 2

    def test_has_infinite_loop_risk_detects_while(self, verifier):
        assert verifier._has_infinite_loop_risk("while True: pass") is True

    def test_has_infinite_loop_risk_detects_for(self, verifier):
        assert verifier._has_infinite_loop_risk("for i in range(10): pass") is True

    def test_has_infinite_loop_risk_clean(self, verifier):
        assert verifier._has_infinite_loop_risk("def add(a, b): return a + b") is False

    def test_has_infinite_loop_risk_syntax_error(self, verifier):
        assert verifier._has_infinite_loop_risk("def broken(") is False

    def test_sandbox_result_to_dict(self):
        result = SandboxResult(score=0.75, tests_passed=3, tests_total=4, errors=["test_2: failed"], timeout=False)
        d = result.to_dict()
        assert d["score"] == 0.75
        assert d["tests_passed"] == 3
        assert d["tests_total"] == 4
        assert d["errors"] == ["test_2: failed"]
        assert d["timeout"] is False

    def test_infinite_loop_risk_is_logged(self, verifier):
        code = "while True: pass"
        assert verifier._has_infinite_loop_risk(code) is True
