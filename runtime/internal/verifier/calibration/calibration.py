"""Calibration Loop and Fidelity Score (AD-13).

Compares SimVerif predictions with real sandbox executions.
Fidelity Score = 40% accuracy + 40% expert suite pass rate + 20% user feedback.
Auto-disables simulation cells with Fidelity < 0.85.
"""

import logging
import random
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("verifier.calibration")

FIDELITY_THRESHOLD = 0.85
SAMPLING_RATE = 0.05
EXPERT_SUITE_WEIGHT = 0.40
ACCURACY_WEIGHT = 0.40
USER_FEEDBACK_WEIGHT = 0.20


class CalibrationEvent:
    def __init__(self, domain: str, env: str, tool_type: str,
                 simulated_output: str, real_output: str, match: bool):
        self.domain = domain
        self.env = env
        self.tool_type = tool_type
        self.simulated_output = simulated_output
        self.real_output = real_output
        self.match = match

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "env": self.env,
            "tool_type": self.tool_type,
            "simulated_output": self.simulated_output,
            "real_output": self.real_output,
            "match": self.match,
        }


class CalibrationRegistry:
    def __init__(self):
        self._events: list[CalibrationEvent] = []
        self._expert_results: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
        self._user_feedback: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
        self._disabled_cells: set[tuple[str, str, str]] = set()

    def should_sample(self, domain: str, env: str, tool_type: str) -> bool:
        key = (domain, env, tool_type)
        if key in self._disabled_cells:
            return True
        return random.random() < SAMPLING_RATE

    def record_event(self, event: CalibrationEvent) -> None:
        self._events.append(event)

    def record_expert_result(self, domain: str, env: str, tool_type: str, passed: bool) -> None:
        self._expert_results[(domain, env, tool_type)].append(passed)

    def record_feedback(self, domain: str, env: str, tool_type: str, positive: bool) -> None:
        self._user_feedback[(domain, env, tool_type)].append(positive)

    def calculate_fidelity(self, domain: str, env: str, tool_type: str) -> float:
        key = (domain, env, tool_type)

        accuracy = self._calculate_accuracy(key)
        expert_pass_rate = self._calculate_expert_pass_rate(key)
        user_positive_ratio = self._calculate_user_feedback(key)

        score = (
            accuracy * ACCURACY_WEIGHT
            + expert_pass_rate * EXPERT_SUITE_WEIGHT
            + user_positive_ratio * USER_FEEDBACK_WEIGHT
        )

        return round(score, 4)

    def _calculate_accuracy(self, key: tuple) -> float:
        relevant = [e for e in self._events
                    if (e.domain, e.env, e.tool_type) == key]
        if not relevant:
            return 1.0
        matches = sum(1 for e in relevant if e.match)
        return matches / len(relevant)

    def _calculate_expert_pass_rate(self, key: tuple) -> float:
        results = self._expert_results.get(key, [])
        if not results:
            return 1.0
        return sum(results) / len(results)

    def _calculate_user_feedback(self, key: tuple) -> float:
        feedback = self._user_feedback.get(key, [])
        if not feedback:
            return 1.0
        return sum(feedback) / len(feedback)

    def run_calibration(self) -> list[dict]:
        all_keys = set()

        for e in self._events:
            all_keys.add((e.domain, e.env, e.tool_type))

        for key in self._expert_results:
            all_keys.add(key)
        for key in self._user_feedback:
            all_keys.add(key)

        results = []
        for key in all_keys:
            domain, env, tool_type = key
            fidelity = self.calculate_fidelity(domain, env, tool_type)

            if fidelity < FIDELITY_THRESHOLD:
                self._disabled_cells.add(key)
                logger.warning("Simulation disabled for %s/%s/%s — Fidelity: %.4f",
                               domain, env, tool_type, fidelity)

            results.append({
                "domain": domain,
                "env": env,
                "tool_type": tool_type,
                "fidelity_score": fidelity,
                "disabled": key in self._disabled_cells,
                "threshold": FIDELITY_THRESHOLD,
            })

        return results

    def is_disabled(self, domain: str, env: str, tool_type: str) -> bool:
        return (domain, env, tool_type) in self._disabled_cells

    def get_status(self) -> dict:
        return {
            "total_events": len(self._events),
            "disabled_cells": list(self._disabled_cells),
            "statuses": [
                {
                    "domain": d,
                    "env": e,
                    "tool": t,
                    "disabled": (d, e, t) in self._disabled_cells,
                }
                for d, e, t in sorted(self._disabled_cells)
            ],
        }
