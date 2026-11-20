"""Transparent offline cost-sensitive action-value learner, NOT reinforcement learning.

Fit ONLY on the internal 'policy' partition. Inference features NEVER include true labels
or strong model probabilities. Strong logits are retrieved only after CALL_STRONG.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from onco_agent.agents.policy import ActionTrace

ACTIONS = ('ACCEPT_FAST', 'CALL_STRONG', 'REVIEW')


def state_features(fast_probs, quality):
    p = np.asarray(fast_probs, dtype=float)
    if p.ndim != 1 or not np.isclose(p.sum(), 1, atol=1e-5):
        raise ValueError('Normalized fast probabilities required')
    top = np.sort(p)[-2:]
    second = float(top[-2]) if len(top) >= 2 else 0.
    entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, None))))
    return np.array([float(top[-1]), float(top[-1]-second), entropy,
                     float(quality.get('brightness', .5)),
                     np.log1p(max(0., float(quality.get('edge_variance', 0)))),
                     float(quality.get('saturation', 0))], dtype=float)


def _reward(label, prediction, cost, wrong_penalty):
    return (1. if label == prediction else -wrong_penalty) - cost


@dataclass
class LearnedPolicy:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    max_strong_calls: int = 1

    def choose(self, fast_probs, quality):
        state = state_features(fast_probs, quality)
        features = np.append((state - self.mean) / self.std, 1.)
        utility = features @ self.coef
        if not self.max_strong_calls:
            utility[1] = -np.inf
        return ACTIONS[int(np.argmax(utility))]

    def run(self, fast_probs, strong_predict=None, quality=None):
        fast = np.asarray(fast_probs, dtype=float)
        action = self.choose(fast, quality or {})
        if action == 'REVIEW':
            return ActionTrace(('fast_predict', 'review'), float(fast.max()), True,
                               None, 0, 'learned_review')
        if action == 'CALL_STRONG':
            if strong_predict is None:
                raise ValueError('Learned policy requires callable strong tool')
            strong = np.asarray(strong_predict(), dtype=float)
            if strong.shape != fast.shape or not np.isclose(strong.sum(), 1, atol=1e-5):
                raise ValueError('Invalid strong output')
            return ActionTrace(('fast_predict', 'strong_predict'), float(strong.max()),
                               False, int(strong.argmax()), 1, 'learned_strong')
        return ActionTrace(('fast_predict',), float(fast.max()), False,
                           int(fast.argmax()), 0, 'learned_fast')

    def save(self, dest, costs):
        p = Path(dest); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'method': 'full_information_cost_sensitive_ridge',
                                 'is_reinforcement_learning': False,
                                 'mean': self.mean.tolist(), 'std': self.std.tolist(),
                                 'coef': self.coef.tolist(), 'costs': costs,
                                 'max_strong_calls': self.max_strong_calls}, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(np.asarray(data['mean']), np.asarray(data['std']),
                   np.asarray(data['coef']), int(data['max_strong_calls']))


def fit_policy(fast_probs, strong_probs, labels, quality_rows, *,
               strong_cost=.10, review_cost=.25, wrong_penalty=2., ridge=.5):
    fast = np.asarray(fast_probs)
    strong = np.asarray(strong_probs)
    y = np.asarray(labels, dtype=int)
    if fast.shape != strong.shape or len(fast) != len(y) or not len(y):
        raise ValueError('Expected paired probabilities with aligned labels')
    x = np.asarray([state_features(fast[i], quality_rows[i]) for i in range(len(y))])
    rewards = np.column_stack([
        np.asarray([_reward(int(y[i]), int(fast[i].argmax()), 0., wrong_penalty)
                    for i in range(len(y))]),
        np.asarray([_reward(int(y[i]), int(strong[i].argmax()), strong_cost, wrong_penalty)
                    for i in range(len(y))]),
        np.full(len(y), -review_cost)
    ])
    mean = x.mean(0); std = np.where(x.std(0) < 1e-8, 1., x.std(0))
    design = np.column_stack([(x-mean)/std, np.ones(len(x))])
    penalty = np.diag([ridge]*x.shape[1]+[0.])
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ rewards)
    return LearnedPolicy(mean, std, coef), {'strong_cost': strong_cost,
        'review_cost': review_cost, 'wrong_penalty': wrong_penalty,
        'ridge': ridge, 'training_examples': len(y)}
