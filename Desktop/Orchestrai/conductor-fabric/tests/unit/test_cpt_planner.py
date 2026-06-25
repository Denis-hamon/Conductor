"""Tests for CptPlanner — model wrapper with confidence threshold."""

from unittest.mock import patch

import pytest

from training.phase0_cpt.model.cpt_planner import CptPlanner, CptResult


class TestCptPlannerAvailable:
    def test_not_available_with_no_path(self):
        planner = CptPlanner(model_path="")
        assert not planner.available

    def test_not_available_with_nonexistent_path(self):
        planner = CptPlanner(model_path="/nonexistent/path")
        assert not planner.available

    @patch.dict("os.environ", {"CPT_MODEL_PATH": "/some/path"})
    @patch("os.path.exists", return_value=True)
    def test_available_with_env_var(self, mock_exists):
        planner = CptPlanner(model_path="")
        assert planner.available

    @patch.dict("os.environ", {"CPT_MODEL_PATH": "/some/path"})
    @patch("os.path.exists", return_value=False)
    def test_available_with_env_var_but_missing_path(self, mock_exists):
        planner = CptPlanner(model_path="")
        assert not planner.available


class TestCptPlannerFallback:
    def test_fallback_when_model_not_loaded(self):
        planner = CptPlanner(model_path="")
        result = planner.generate("Write code")
        assert isinstance(result, CptResult)
        assert result.fallback_used
        assert result.model_name == "heuristic"
        assert "steps" in result.plan

    def test_fallback_plan_structure(self):
        planner = CptPlanner(model_path="")
        result = planner.generate("Solve math problem: 2+2")
        assert result.fallback_used
        plan = result.plan
        assert "domain" in plan
        assert "steps" in plan
        assert "verification_gates" in plan
        assert "stop_condition" in plan


class TestCptPlannerParsePlan:
    def test_parse_valid_json(self):
        planner = CptPlanner(model_path="")
        text = '{"domain": "code", "confidence": 0.85, "steps": [], "verification_gates": [], "stop_condition": "on_complete"}'
        plan = planner._parse_plan(text)
        assert plan["domain"] == "code"
        assert plan["confidence"] == 0.85

    def test_parse_json_with_extra_text(self):
        planner = CptPlanner(model_path="")
        text = 'Here is your plan:\n{"domain": "rag", "confidence": 0.72}\nEnd'
        plan = planner._parse_plan(text)
        assert plan["domain"] == "rag"
        assert plan["confidence"] == 0.72

    def test_parse_invalid_text(self):
        planner = CptPlanner(model_path="")
        plan = planner._parse_plan("not json at all")
        assert plan["confidence"] == 0.0
        assert plan["domain"] == "general"

    def test_parse_empty_text(self):
        planner = CptPlanner(model_path="")
        plan = planner._parse_plan("")
        assert plan["confidence"] == 0.0


class TestCptResult:
    def test_cpt_result_fields(self):
        result = CptResult(
            plan={"domain": "code"},
            confidence=0.85,
            model_name="qwen-0.5b-cpt",
            fallback_used=False,
        )
        assert result.plan["domain"] == "code"
        assert result.confidence == 0.85
        assert not result.fallback_used

    def test_fallback_result_fields(self):
        result = CptResult(
            plan={"domain": "general"},
            confidence=0.0,
            model_name="heuristic",
            fallback_used=True,
        )
        assert result.fallback_used
        assert result.model_name == "heuristic"
