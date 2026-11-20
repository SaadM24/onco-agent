"""Replay logged model probabilities to benchmark tool policies without test-label leakage."""
from __future__ import annotations
import numpy as np
from .metrics import selective_metrics
from onco_agent.agents.policy import Policy


def evaluate_policy(fast_probs, labels, policy: Policy, strong_probs=None, quality_flags=None):
    fast = np.asarray(fast_probs, dtype=float)
    y = np.asarray(labels)
    strong = None if strong_probs is None else np.asarray(strong_probs, dtype=float)
    quality = np.zeros(len(y), dtype=bool) if quality_flags is None else np.asarray(quality_flags, dtype=bool)
    if len(fast) != len(y) or len(quality) != len(y) or (strong is not None and strong.shape != fast.shape):
        raise ValueError("Inputs not aligned")
    traces = []
    pred = np.zeros_like(fast)
    abstain = []
    for i in range(len(y)):
        callback = (lambda i=i: strong[i]) if strong is not None else None
        t = policy.run(fast[i], strong_predict=callback, quality_flag=bool(quality[i]))
        traces.append(t)
        abstain.append(t.abstained)
        pred[i] = strong[i] if t.strong_tool_calls else fast[i]
    metrics = selective_metrics(pred, y, abstain)
    metrics.update({"strong_tool_calls": int(sum(t.strong_tool_calls for t in traces)),
                    "strong_calls_per_image": float(np.mean([t.strong_tool_calls for t in traces]))})
    return metrics, traces
