"""Simple slope regressions (y on x with an intercept) through the influence-value machinery,
for the SPF benchmark regressions."""

from __future__ import annotations

import math

import numpy as np

from ewh.estimators.eor import _demean, ratio_estimate
from ewh.estimators.types import CovSpec, Draws, Estimate


def slope(
    y: np.ndarray,
    x: np.ndarray,
    clusters: dict[str, np.ndarray],
    cov: CovSpec,
    cell: np.ndarray | None = None,
    label: str = "",
) -> Estimate:
    """OLS slope of y on x with intercepts per `cell` (one intercept if None)."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    cell = np.zeros(n) if cell is None else cell[ok]
    names = tuple(clusters)
    mat = (
        np.column_stack([np.asarray(clusters[k])[ok] for k in names]) if names else np.zeros((n, 0))
    )
    d = Draws(np.zeros((n, 1)), np.zeros((n, 1)), np.zeros(n), cell, mat, names)
    xt, yt = _demean(x, cell), _demean(y, cell)
    return ratio_estimate(xt * yt, xt**2, d, cov, label=label, weak=False)


def nw_auto_lags(T: int) -> int:
    """Newey-West (1994) plug-in rule of thumb for the Bartlett kernel: floor(4 (T/100)^(2/9))."""
    return int(math.floor(4 * (T / 100) ** (2 / 9)))
