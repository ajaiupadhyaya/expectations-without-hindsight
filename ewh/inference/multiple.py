"""Holm step-down correction and minimum detectable effects."""

from __future__ import annotations

from scipy import stats


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm (1979): reject the i-th smallest p-value while p_(i) <= alpha / (m - i + 1)."""
    m = len(pvals)
    reject = dict.fromkeys(pvals, False)
    for i, (name, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        if p <= alpha / (m - i):
            reject[name] = True
        else:
            break
    return reject


def holm_adjusted(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (monotone)."""
    m = len(pvals)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Two-sided minimum detectable effect: (z_{1-alpha/2} + z_power) * se (2.80 se at 5%/80%)."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)
