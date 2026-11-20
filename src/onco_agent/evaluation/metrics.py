"""Classification and selective-risk metrics; no clinical interpretation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, log_loss


def expected_calibration_error(probs, labels, n_bins=10):
    p = np.asarray(probs)
    labels = np.asarray(labels)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    total = 0.0
    for a, b in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        group = (conf >= a) & ((conf <= b) if b == 1 else (conf < b))
        if group.any():
            total += float(group.mean()) * abs(float((pred[group] == labels[group]).mean()) - float(conf[group].mean()))
    return float(total)


def classification_metrics(probs, labels):
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 2 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Nonempty, aligned arrays required")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Input must contain normalized probabilities")
    pred = p.argmax(axis=1)
    n = p.shape[1]
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(n), zero_division=0)
    one_hot = np.eye(n)[y]
    safe = np.clip(p, 1e-12, 1.0)
    safe = safe / safe.sum(axis=1, keepdims=True)
    return {"samples": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=np.arange(n), average="macro", zero_division=0)),
            "nll": float(log_loss(y, safe, labels=np.arange(n))),
            "brier_multiclass": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
            "ece_10bin": expected_calibration_error(p, y),
            "per_class": {str(i): {"precision": float(prec[i]), "recall": float(rec[i]),
                                    "f1": float(f1[i]), "support": int(support[i])} for i in range(n)}}


def selective_metrics(probs, labels, abstain):
    p = np.asarray(probs)
    y = np.asarray(labels)
    mask = ~np.asarray(abstain, dtype=bool)
    if len(mask) != len(y) or len(y) != len(p):
        raise ValueError("Inputs not aligned")
    accepted = int(mask.sum())
    return {"coverage": float(mask.mean()), "abstention_rate": float(1 - mask.mean()),
            "accepted": accepted,
            "selective_error": float((p.argmax(axis=1)[mask] != y[mask]).mean()) if accepted else None,
            "accepted_accuracy": float((p.argmax(axis=1)[mask] == y[mask]).mean()) if accepted else None}
