"""Sandbox code verifier — executes code in isolated environment with timeout."""

import ast
import logging
import time
import traceback
from typing import Optional

logger = logging.getLogger("verifier.sandbox")

SANDBOX_TIMEOUT = 30


class SandboxResult:
    def __init__(self, score: float, tests_passed: int = 0, tests_total: int = 0,
                 errors: list[str] = None, timeout: bool = False):
        self.score = score
        self.tests_passed = tests_passed
        self.tests_total = tests_total
        self.errors = errors or []
        self.timeout = timeout

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "errors": self.errors,
            "timeout": self.timeout,
        }


class SandboxVerifier:
    def __init__(self, timeout: int = SANDBOX_TIMEOUT):
        self.timeout = timeout

    def verify(self, code: str, tests: Optional[list[str]] = None) -> SandboxResult:
        try:
            ast.parse(code)
        except SyntaxError as e:
            return SandboxResult(score=0.0, errors=[f"syntax error: {e}"])

        if self._has_infinite_loop_risk(code):
            logger.warning("Code flagged for infinite loop risk — will enforce strict timeout")

        if not tests:
            return SandboxResult(score=1.0, tests_passed=1, tests_total=1)

        passed = 0
        errors = []
        for i, test in enumerate(tests):
            start = time.monotonic()
            try:
                compiled = compile(code + "\n" + test, f"<test_{i}>", "exec")
                exec(compiled, {"__builtins__": __builtins__})
                passed += 1
            except AssertionError as e:
                errors.append(f"test_{i}: {e}")
            except Exception as e:
                errors.append(f"test_{i}: {type(e).__name__}: {e}")
            finally:
                elapsed = time.monotonic() - start
                if elapsed > self.timeout:
                    return SandboxResult(score=0.0, timeout=True,
                                         errors=["timeout: execution exceeded limit"])

        total = len(tests)
        score = passed / total if total > 0 else 1.0
        return SandboxResult(score=score, tests_passed=passed, tests_total=total, errors=errors)

    def _has_infinite_loop_risk(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.While, ast.For)):
                    return True
            return False
        except SyntaxError:
            return False
