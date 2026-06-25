"""Tests for CalibrationRegistry — fidelity score, sampling, auto-disable."""

import pytest

from runtime.internal.verifier.calibration.calibration import (
    CalibrationRegistry, CalibrationEvent, FIDELITY_THRESHOLD,
)


@pytest.fixture
def registry():
    return CalibrationRegistry()


class TestCalibrationRegistry:
    def test_initial_fidelity_is_1_0(self, registry):
        score = registry.calculate_fidelity("code", "swe", "python")
        assert score == 1.0

    def test_record_event_and_calculate_accuracy(self, registry):
        registry.record_event(CalibrationEvent("code", "swe", "python", "out1", "out1", match=True))
        registry.record_event(CalibrationEvent("code", "swe", "python", "out2", "out2", match=True))
        registry.record_event(CalibrationEvent("code", "swe", "python", "out3", "out3_wrong", match=False))
        score = registry.calculate_fidelity("code", "swe", "python")
        assert score < 1.0
        assert score > 0.8

    def test_expert_results_affect_fidelity(self, registry):
        registry.record_expert_result("rag", "mcp", "search", True)
        registry.record_expert_result("rag", "mcp", "search", True)
        registry.record_expert_result("rag", "mcp", "search", False)
        score = registry.calculate_fidelity("rag", "mcp", "search")
        assert score < 1.0

    def test_user_feedback_affects_fidelity(self, registry):
        registry.record_feedback("code", "terminal", "bash", True)
        registry.record_feedback("code", "terminal", "bash", False)
        score = registry.calculate_fidelity("code", "terminal", "bash")
        assert score < 1.0

    def test_run_calibration_disables_low_fidelity(self, registry):
        registry.record_event(CalibrationEvent("code", "swe", "python", "a", "b", match=False))
        registry.record_event(CalibrationEvent("code", "swe", "python", "c", "d", match=False))
        registry.record_event(CalibrationEvent("code", "swe", "python", "e", "f", match=False))
        results = registry.run_calibration()
        code_result = [r for r in results if r["domain"] == "code"]
        assert len(code_result) == 1
        assert code_result[0]["disabled"] is True

    def test_is_disabled_after_calibration(self, registry):
        registry.record_event(CalibrationEvent("code", "swe", "python", "a", "b", match=False))
        registry.run_calibration()
        assert registry.is_disabled("code", "swe", "python") is True

    def test_not_disabled_by_default(self, registry):
        assert registry.is_disabled("code", "swe", "python") is False

    def test_disabled_cell_always_samples(self, registry):
        registry.record_event(CalibrationEvent("code", "swe", "python", "a", "b", match=False))
        registry.run_calibration()
        assert registry.should_sample("code", "swe", "python") is True

    def test_get_status(self, registry):
        registry.record_event(CalibrationEvent("rag", "mcp", "tool", "a", "b", match=False))
        registry.run_calibration()
        status = registry.get_status()
        assert status["total_events"] == 1
        assert len(status["disabled_cells"]) >= 0

    def test_calibration_event_to_dict(self):
        event = CalibrationEvent("code", "swe", "python", "sim_out", "real_out", match=True)
        d = event.to_dict()
        assert d["domain"] == "code"
        assert d["match"] is True
