"""Variance of a mean of influence values under clustering, two-way clustering and
Driscoll-Kraay; weak-instrument-robust Fieller sets; fixed-b critical values.

Convention: an estimator supplies psi (n,) with b_hat - b ~ mean(psi); `var_of_mean` returns
Var(mean(psi)).
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy import stats

from ewh.estimators.types import CovSpec, Draws


def _oneway(psi: np.ndarray, g: np.ndarray) -> tuple[float, int]:
    n = len(psi)
    _, idx = np.unique(g, return_inverse=True)
    sums = np.bincount(idx, weights=psi)
    G = len(sums)
    return float((sums**2).sum() / n**2 * G / (G - 1)), G


def _bartlett(series: np.ndarray, lags: int) -> float:
    """Sum of autocovariances of a (T,) or (T, m) series with Bartlett weights (no centring:
    influence values already have mean zero)."""
    s = series if series.ndim == 2 else series[:, None]
    out = s.T @ s
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = s[lag:].T @ s[:-lag]
        out = out + w * (g + g.T)
    return out if series.ndim == 2 else float(out[0, 0])


def _time_sums(psi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Sum influence values within each period, on a complete integer time grid."""
    t = np.asarray(t, dtype=int)
    T = t.max() - t.min() + 1
    sums = np.zeros((T,) + psi.shape[1:])
    np.add.at(sums, t - t.min(), psi)
    return sums


def var_of_mean(psi: np.ndarray, d: Draws, spec: CovSpec) -> tuple[float, tuple[int, ...]]:
    """Var(mean(psi)) for units of `d` (same order), with the cluster counts used."""
    n = len(psi)
    if spec.kind == "cr1":
        v, G = _oneway(psi, d.column(spec.clusters[0]))
        return v, (G,)
    if spec.kind == "twoway":
        g1, g2 = d.column(spec.clusters[0]), d.column(spec.clusters[1])
        v1, G1 = _oneway(psi, g1)
        v2, G2 = _oneway(psi, g2)
        _, i1 = np.unique(g1, return_inverse=True)
        _, i2 = np.unique(g2, return_inverse=True)
        v12, _ = _oneway(psi, i1 * (i2.max() + 1) + i2)
        v = v1 + v2 - v12
        return (v if v > 0 else max(v1, v2)), (G1, G2)
    if spec.kind in ("dk", "nw"):
        assert spec.time is not None and spec.lags is not None
        sums = _time_sums(psi, d.column(spec.time))
        return _bartlett(sums, spec.lags) / n**2, (len(sums),)
    raise ValueError(spec.kind)


def var_matrix_of_means(psis: np.ndarray, d: Draws, spec: CovSpec) -> np.ndarray:
    """2x2 (or m x m) covariance of the means of several influence columns (n, m)."""
    m = psis.shape[1]
    out = np.empty((m, m))
    for i in range(m):
        out[i, i] = var_of_mean(psis[:, i], d, spec)[0]
    for i in range(m):
        for j in range(i + 1, m):
            vs = var_of_mean(psis[:, i] + psis[:, j], d, spec)[0]
            out[i, j] = out[j, i] = (vs - out[i, i] - out[j, j]) / 2
    return out


def fieller(
    abar: float, dbar: float, vaa: float, vad: float, vdd: float, level: float = 0.95
) -> tuple[tuple[float, float], str]:
    """Confidence set for b = A/D: {b0 : (abar - b0 dbar)^2 <= q Var(abar - b0 dbar)}.

    Returns ((lo, hi), kind). kind "bounded": [lo, hi]; "complement": (-inf, lo] U [hi, inf);
    "unbounded": the whole line.
    """
    q = stats.chi2.ppf(level, 1)
    A = dbar**2 - q * vdd
    B = -2 * (abar * dbar - q * vad)
    C = abar**2 - q * vaa
    disc = B**2 - 4 * A * C
    if A > 0:
        r = np.sqrt(max(disc, 0.0))
        return ((-B - r) / (2 * A), (-B + r) / (2 * A)), "bounded"
    if disc > 0:
        r = np.sqrt(disc)
        lo, hi = sorted(((-B - r) / (2 * A), (-B + r) / (2 * A)))
        return (lo, hi), "complement"
    return (-np.inf, np.inf), "unbounded"


@cache
def fixed_b_critical(b: float, level: float = 0.95, n_sim: int = 20000, T: int = 400) -> float:
    """Two-sided fixed-b critical value for a Bartlett-kernel t-test with bandwidth
    lags = b*T (Kiefer-Vogelsang asymptotics), obtained by simulation of i.i.d. normal data.
    Returns the normal value when b = 0."""
    if b <= 0:
        return float(stats.norm.ppf(0.5 + level / 2))
    rng = np.random.default_rng(12345)
    lags = max(1, int(round(b * T)))
    tstats = np.empty(n_sim)
    for s in range(n_sim):
        e = rng.standard_normal(T)
        u = e - e.mean()
        v = _bartlett(u, lags) / T**2
        tstats[s] = e.mean() / np.sqrt(v)
    return float(np.quantile(np.abs(tstats), level))


def normal_p(t: float) -> float:
    return float(2 * stats.norm.sf(abs(t)))
