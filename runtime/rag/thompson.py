"""Layer 6: Thompson Sampling feedback loop for self-improving retrieval.

Maintains a Beta distribution for each hyperparameter configuration.
After each retrieval, updates priors based on success/failure.
Adjusts alpha (dense/sparse blend), top_k, rerank_k in real-time.

Reference: HydraDB/Cortex uses Thompson Sampling for adaptive retrieval.
  - Bandit-based approach to balance exploration vs exploitation
  - Converges to optimal retrieval parameters for each query type
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rag.thompson")

ALPHA_GRID = [0.3, 0.5, 0.7, 0.9]
TOP_K_GRID = [5, 10, 20, 30]
RERANK_K_GRID = [3, 5, 10]


@dataclass
class ParamConfig:
    alpha: float = 0.7
    top_k: int = 20
    rerank_k: int = 10

    def key(self) -> str:
        return f"a={self.alpha:.1f}_k={self.top_k}_r={self.rerank_k}"

    @staticmethod
    def from_key(key: str):
        parts = key.split("_")
        alpha = float(parts[0].split("=")[1])
        top_k = int(parts[1].split("=")[1])
        rerank_k = int(parts[2].split("=")[1])
        return ParamConfig(alpha, top_k, rerank_k)


class ThompsonSampler:
    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path) if state_path else None
        self.priors: dict[str, tuple[int, int]] = {}
        self._init_priors()

        if self.state_path and self.state_path.exists():
            self.load()

    def _init_priors(self):
        for a in ALPHA_GRID:
            for t in TOP_K_GRID:
                for r in RERANK_K_GRID:
                    cfg = ParamConfig(alpha=a, top_k=t, rerank_k=r)
                    self.priors[cfg.key()] = (1, 1)

    def sample(self) -> ParamConfig:
        best_key = None
        best_score = -float("inf")

        for key, (success, failure) in self.priors.items():
            score = random.betavariate(success + 1, failure + 1)
            if score > best_score:
                best_score = score
                best_key = key

        return ParamConfig.from_key(best_key)

    def update(self, config: ParamConfig, success: bool):
        key = config.key()
        if key not in self.priors:
            self.priors[key] = (1, 1)

        s, f = self.priors[key]
        if success:
            self.priors[key] = (s + 1, f)
        else:
            self.priors[key] = (s, f + 1)

        logger.debug("Thompson update: %s → success=%s (α=%d, β=%d)",
                      key, success, *self.priors[key])
        self.save()

    def best_config(self) -> ParamConfig:
        best_key = max(self.priors, key=lambda k: self._expected_success(k))
        return ParamConfig.from_key(best_key)

    def _expected_success(self, key: str) -> float:
        s, f = self.priors[key]
        return s / (s + f)

    def save(self):
        if not self.state_path:
            return
        data = {k: list(v) for k, v in self.priors.items()}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        with open(self.state_path) as f:
            data = json.load(f)
        for k, v in data.items():
            self.priors[k] = tuple(v)
        logger.info("Loaded Thompson state: %d configurations", len(self.priors))

    def top_configs(self, n: int = 5) -> list[tuple[ParamConfig, float]]:
        scored = [(ParamConfig.from_key(k), self._expected_success(k))
                   for k in self.priors]
        scored.sort(key=lambda x: -x[1])
        return scored[:n]
