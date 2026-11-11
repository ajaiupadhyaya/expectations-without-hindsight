"""Implied persistence (AKLMT) and the noise share."""

from __future__ import annotations

import numpy as np

from ewh.estimators.eor import _demean, common_sample, ratio_estimate
from ewh.estimators.types import CovSpec, Draws, Estimate


def implied_persistence(d: Draws, x_last: np.ndarray, cov: CovSpec, min_valid: int = 8) -> Estimate:
    """Slope of the mean one-step forecast (date t) on the last shown value, with cell
    intercepts. Draw noise is on the left-hand side, so no correction is needed.

    `x_last` is aligned with `d` (one value per unit)."""
    keep = np.isin(d.uid, common_sample(d, min_valid).uid)
    c = d.subset(keep)
    f = np.nanmean(c.f_t, 1)
    xt = _demean(x_last[keep], c.cell)
    ft = _demean(f, c.cell)
    return ratio_estimate(xt * ft, xt**2, c, cov, label="implied_persistence", weak=False)


def noise_share(d: Draws, which: str = "t") -> float:
    """Within-prompt draw variance over the total variance of single draws."""
    f = d.f_t if which == "t" else d.f_tm1
    within = np.nanmean(np.nanvar(f, 1, ddof=1))
    total = np.nanvar(f[np.isfinite(f)], ddof=1)
    return float(within / total) if total > 0 else 0.0
