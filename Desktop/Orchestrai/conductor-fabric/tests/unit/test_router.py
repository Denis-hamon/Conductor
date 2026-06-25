"""Tests for heuristic request router — domain classification."""

import pytest

from conductor.router import classify_request, _score_patterns, RouteResult


class TestScorePatterns:
    def test_all_patterns_match(self):
        patterns = [r"hello", r"world"]
        score = _score_patterns("hello world", patterns)
        assert score == 1.0

    def test_partial_match(self):
        patterns = [r"hello", r"world", r"foo"]
        score = _score_patterns("hello there", patterns)
        assert score == pytest.approx(1 / 3)

    def test_no_match(self):
        patterns = [r"hello", r"world"]
        score = _score_patterns("goodbye", patterns)
        assert score == 0.0

    def test_empty_patterns(self):
        score = _score_patterns("anything", [])
        assert score == 0.0


class TestClassifyRequest:
    def test_code_python_request(self):
        result = classify_request("Write a Python function to sort a list")
        assert result.domain == "code"

    def test_code_with_debug_and_error(self):
        result = classify_request("I need to debug this function — it throws an error when I run the code")
        assert result.domain == "code"

    def test_code_with_function_keyword(self):
        result = classify_request("Write a Python function to sort a list")
        assert result.domain == "code"

    def test_rag_with_legal_content(self):
        result = classify_request("Summarize section 4 of the contract according to the document")
        assert result.domain == "rag"

    def test_rag_with_source_reference(self):
        result = classify_request("Find the clause about termination from the source file")
        assert result.domain == "rag"

    def test_reason_math_request(self):
        result = classify_request("Solve this equation: 2x + 5 = 15")
        assert result.domain == "reason"

    def test_reason_analysis(self):
        result = classify_request("Analyze the pros and cons of microservices compare the options")
        assert result.domain == "reason"

    def test_reason_explain_causality(self):
        result = classify_request("Explain how does this work and why does it matter")
        assert result.domain == "reason"

    def test_mcp_tool_call(self):
        result = classify_request("call the function and execute a command using the mcp protocol")
        assert result.domain == "mcp"

    def test_mcp_with_connector(self):
        result = classify_request("use the connector tool and execute a query via the API mcp")
        assert result.domain == "mcp"

    def test_mcp_with_execute(self):
        result = classify_request("call the API endpoint and execute a command")
        assert result.domain == "mcp"

    def test_general_question(self):
        result = classify_request("What is the capital of France?")
        assert result.domain == "general"
        assert result.confidence < 0.3

    def test_general_greeting(self):
        result = classify_request("Hello, how are you?")
        assert result.domain == "general"

    def test_returns_route_result_tuple(self):
        result = classify_request("Write code")
        assert isinstance(result, RouteResult)
        assert hasattr(result, "domain")
        assert hasattr(result, "confidence")
        assert hasattr(result, "reason")

    def test_case_insensitive(self):
        result_lower = classify_request("WRITE A PYTHON FUNCTION")
        result_upper = classify_request("write a python function")
        assert result_lower.domain == result_upper.domain

    def test_code_beats_general_when_both_match(self):
        result = classify_request("Write a Python function to calculate fibonacci")
        assert result.domain == "code"
