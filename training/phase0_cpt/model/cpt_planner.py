"""CPT-based planner — uses a trained small model for plan generation.

When confidence ≥ threshold, the CPT model's plan is used.
Otherwise, falls back to the heuristic planner.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("phase0_cpt.planner")


@dataclass
class CptResult:
    plan: dict[str, Any]
    confidence: float
    model_name: str
    fallback_used: bool


PROMPT_TEMPLATE = """Given the following user request, generate a workflow plan.

Request: {query}

Respond with a JSON object containing:
- "domain": the domain (code, rag, reason, general, mcp)
- "steps": an array of execution steps, each with "type", "agent", "role", "budget_tokens"
- "verification_gates": an array of verification gates, each with "type" and "weight"
- "stop_condition": when to stop ("on_complete")
- "confidence": a float between 0 and 1 indicating your confidence in this plan

JSON:
"""


class CptPlanner:
    def __init__(
        self,
        model_path: str = "",
        threshold: float = 0.7,
        max_tokens: int = 1024,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def available(self) -> bool:
        path = self.model_path or os.environ.get("CPT_MODEL_PATH", "")
        if not path:
            return False
        if not os.path.exists(path):
            return False
        return True

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            path = os.environ.get("CPT_MODEL_PATH", self.model_path) or ""
            if not path:
                raise FileNotFoundError("CPT model path not configured")

            logger.info("Loading CPT model from %s on %s", path, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            kwargs = {}
            if self.device and self.device != "cpu":
                kwargs["device_map"] = self.device
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                **kwargs,
            )
            logger.info("CPT model loaded: %s", path)
        except Exception:
            logger.exception("Failed to load CPT model")
            self._model = None
            self._tokenizer = None

    def generate(self, content: str) -> CptResult:
        self._lazy_load()
        if self._model is None or self._tokenizer is None:
            logger.warning("CPT model not available, using fallback")
            return self._fallback(content)

        safe_content = content.replace("{", "{{").replace("}", "}}")
        prompt = PROMPT_TEMPLATE.format(query=safe_content)
        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )

            generated = self._tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            plan = self._parse_plan(generated)
            confidence = max(0.0, min(1.0, plan.get("confidence", 0.0)))

            if confidence >= self.threshold:
                return CptResult(
                    plan=plan,
                    confidence=confidence,
                    model_name=os.path.basename(self.model_path or "cpt-model"),
                    fallback_used=False,
                )

            logger.info(
                "CPT confidence %.2f < threshold %.2f, falling back",
                confidence,
                self.threshold,
            )
            return self._fallback(content, confidence)

        except Exception:
            logger.exception("CPT model inference failed")
            return self._fallback(content)

    def _parse_plan(self, text: str) -> dict[str, Any]:
        import re
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass
        return {
            "domain": "general",
            "confidence": 0.0,
            "steps": [],
            "verification_gates": [],
            "stop_condition": "on_complete",
        }

    def _fallback(self, content: str, model_confidence: float = 0.0) -> CptResult:
        from conductor.planner import WorkflowPlanner

        planner = WorkflowPlanner()
        domain = "general"
        plan = planner.generate(domain, content)
        return CptResult(
            plan=plan,
            confidence=model_confidence,
            model_name="heuristic",
            fallback_used=True,
        )
