"""Stateful constrained policies. Labels MUST NOT be in policy inputs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class ActionTrace:
    actions: tuple[str, ...]
    confidence: float
    abstained: bool
    predicted_class: int | None
    strong_tool_calls: int
    reason: str


def _validate_probs(p):
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1 or not len(arr) or not np.isclose(arr.sum(), 1, atol=1e-5) or np.any(arr < 0):
        raise ValueError("Must provide normalized 1-D class probabilities")
    return arr


class Policy:
    """Rule policy or non-agentic fast baseline with a per-image tool budget."""
    def __init__(self, mode: str = "rule", fast_threshold: float = 0.85,
                 review_threshold: float = 0.65, max_strong_calls: int = 1):
        if mode not in {"rule", "fast", "fixed_abstain"}:
            raise ValueError("Unsupported mode")
        if not (0 <= review_threshold <= fast_threshold <= 1):
            raise ValueError("Expect 0 <= review_threshold <= fast_threshold <= 1")
        if max_strong_calls not in (0, 1):
            raise ValueError("max_strong_calls must be 0 or 1")
        self.mode, self.fast_threshold = mode, fast_threshold
        self.review_threshold, self.max_strong_calls = review_threshold, max_strong_calls

    def run(self, fast_probs, strong_predict: Callable[[], np.ndarray] | None = None,
            quality_flag: bool = False):
        fast = _validate_probs(fast_probs)
        actions = ["fast_predict"]
        conf = float(fast.max())
        if self.mode == "fast":
            return ActionTrace(tuple(actions), conf, False, int(fast.argmax()), 0, "fast_only")
        if self.mode == "fixed_abstain":
            abstain = conf < self.fast_threshold
            return ActionTrace(tuple(actions + (["review"] if abstain else [])), conf,
                               abstain, None if abstain else int(fast.argmax()), 0, "fixed_threshold")
        current = fast
        calls = 0
        if (quality_flag or conf < self.fast_threshold) and strong_predict is not None and self.max_strong_calls:
            current = _validate_probs(strong_predict())
            if len(current) != len(fast):
                raise ValueError("Strong and fast model class counts differ")
            actions.append("strong_predict")
            calls += 1
        conf = float(current.max())
        disagreement = calls and (current.argmax() != fast.argmax())
        abstain = conf < self.review_threshold or (quality_flag and conf < self.fast_threshold) or bool(disagreement)
        if abstain:
            actions.append("review")
        return ActionTrace(tuple(actions), conf, abstain,
                           None if abstain else int(current.argmax()), calls,
                           "disagreement" if disagreement else "quality_or_uncertainty" if abstain else "accepted")
