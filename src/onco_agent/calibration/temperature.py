"""Fit positive temperature on internal validation logits ONLY."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


def apply_temperature(logits, temperature: float):
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    return softmax(np.asarray(logits, dtype=float) / temperature, axis=1)


def fit_temperature(validation_logits, validation_labels):
    z = np.asarray(validation_logits, dtype=float)
    y = np.asarray(validation_labels, dtype=int)
    if z.ndim != 2 or len(y) != len(z) or len(y) == 0:
        raise ValueError("Validation arrays must be nonempty and aligned")
    if np.any((y < 0) | (y >= z.shape[1])):
        raise ValueError("Labels outside class range")
    def objective(log_t):
        scaled = z / np.exp(log_t)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))
    sol = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not sol.success:
        raise RuntimeError("Temperature optimizer failed")
    return float(np.exp(sol.x))
