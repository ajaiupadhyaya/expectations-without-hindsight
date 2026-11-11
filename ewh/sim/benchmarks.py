"""Benchmark forecasters (handoff 3.3): each maps a shown history (U, T) to an h-step forecast.

All benchmarks go through the same estimation code as the models (plan D10).
"""

from __future__ import annotations

import numpy as np


def rational(hist: np.ndarray, rho: float, h: int) -> np.ndarray:
    """Known-parameter rational forecast (mean 0): rho^h times the last value."""
    return rho**h * hist[:, -1]


def ls_coefficients(hist: np.ndarray, intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-row OLS of x_s on x_{s-1} over the shown history. Returns (a, b), each (U,)."""
    y, x = hist[:, 1:], hist[:, :-1]
    if not intercept:
        b = (x * y).sum(1) / (x * x).sum(1)
        return np.zeros(len(hist)), b
    xm, ym = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
    b = ((x - xm) * (y - ym)).sum(1) / ((x - xm) ** 2).sum(1)
    a = ym[:, 0] - b * xm[:, 0]
    return a, b


def ls_learner(hist: np.ndarray, h: int, intercept: bool = True) -> np.ndarray:
    """Least-squares AR(1) learner given exactly the shown history, iterated h steps."""
    a, b = ls_coefficients(hist, intercept)
    f = hist[:, -1].astype(float)
    for _ in range(h):
        f = a + b * f
    return f


def random_walk(hist: np.ndarray, h: int) -> np.ndarray:
    return hist[:, -1].astype(float)


def uncond_mean(hist: np.ndarray, h: int) -> np.ndarray:
    return hist.mean(1)


echo = random_walk  # "echo the last number": the weak-model failure mode (handoff 1.3)
