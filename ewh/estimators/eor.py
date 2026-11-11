"""Error-on-revision estimators: one family indexed by the number of averaged draws k.

For unit i with valid draws at t and t-1 (counts K_t, K_tm1, per-draw variances s2_t, s2_tm1):
    ebar = x - mean(f_t),  rbar = mean(f_t) - mean(f_tm1).
Noise-free moments (docs/noise_derivation.md):
    N_i = ebar*rbar + s2_t/K_t,   D_i = rbar^2 - s2_t/K_t - s2_tm1/K_tm1.
With cell intercepts, b(k) = sum_c sum_i [e~ r~ + w_c (s2_t/K_t - s2_t/k)]
                           / sum_c sum_i [r~^2 - w_c (s2_t/K_t + s2_tm1/K_tm1 - (s2_t+s2_tm1)/k)],
where e~, r~ are cell-demeaned and w_c = 1 - 1/n_c. k = 1 is the naive (one-draw) limit and
k = inf is the split-sample IV averaged over every split, in closed form (plan change 8).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ewh.estimators.types import CovSpec, Draws, Estimate
from ewh.inference.cov import fieller, normal_p, var_matrix_of_means, var_of_mean

Z95 = 1.959963984540054


@dataclass(frozen=True)
class UnitMoments:
    d: Draws  # the common estimation sample (units with enough valid draws at both dates)
    ebar: np.ndarray
    rbar: np.ndarray
    s2_t: np.ndarray
    s2_tm1: np.ndarray
    k_t: np.ndarray
    k_tm1: np.ndarray


def common_sample(d: Draws, min_valid: int = 8) -> Draws:
    ok = (np.isfinite(d.f_t).sum(1) >= min_valid) & (np.isfinite(d.f_tm1).sum(1) >= min_valid)
    return d.subset(ok)


def unit_moments(d: Draws, min_valid: int = 8) -> UnitMoments:
    c = common_sample(d, min_valid)
    mt, mtm1 = np.nanmean(c.f_t, 1), np.nanmean(c.f_tm1, 1)
    return UnitMoments(
        d=c,
        ebar=c.x - mt,
        rbar=mt - mtm1,
        s2_t=np.nanvar(c.f_t, 1, ddof=1),
        s2_tm1=np.nanvar(c.f_tm1, 1, ddof=1),
        k_t=np.isfinite(c.f_t).sum(1),
        k_tm1=np.isfinite(c.f_tm1).sum(1),
    )


def _demean(v: np.ndarray, cell: np.ndarray) -> np.ndarray:
    _, idx = np.unique(cell, return_inverse=True)
    means = np.bincount(idx, weights=v) / np.bincount(idx)
    return v - means[idx]


def _cell_weight(cell: np.ndarray) -> np.ndarray:
    _, idx, counts = np.unique(cell, return_inverse=True, return_counts=True)
    return 1 - 1 / counts[idx]


def ratio_estimate(
    a: np.ndarray, den: np.ndarray, d: Draws, cov: CovSpec, label: str = "", weak: bool = True
) -> Estimate:
    """b = sum(a)/sum(den) with influence psi = (a - b den)/mean(den)."""
    b = a.sum() / den.sum()
    dbar = den.mean()
    psi = (a - b * den) / dbar
    v, G = var_of_mean(psi, d, cov)
    se = math.sqrt(v)
    est: dict[str, Any] = dict(
        b=float(b),
        se=se,
        ci=(b - Z95 * se, b + Z95 * se),
        p=normal_p(b / se) if se > 0 else float("nan"),
        n_units=d.n,
        n_clusters=G,
        psi=psi,
        uid=d.uid,
        label=label,
    )
    if weak:
        V = var_matrix_of_means(np.column_stack([a - a.mean(), den - dbar]), d, cov)
        est["first_stage_t2"] = float(dbar**2 / V[1, 1]) if V[1, 1] > 0 else float("inf")
        (lo, hi), kind = fieller(a.mean(), dbar, V[0, 0], V[0, 1], V[1, 1])
        est["fieller"], est["fieller_kind"] = (float(lo), float(hi)), kind
    return Estimate(**est)


def family_terms(m: UnitMoments, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit numerator and denominator contributions of b(k)."""
    cell = m.d.cell
    w = _cell_weight(cell)
    et, rt = _demean(m.ebar, cell), _demean(m.rbar, cell)
    inv_k = 0.0 if math.isinf(k) else 1.0 / k
    a = et * rt + w * (m.s2_t / m.k_t - m.s2_t * inv_k)
    den = rt**2 - w * (m.s2_t / m.k_t + m.s2_tm1 / m.k_tm1 - (m.s2_t + m.s2_tm1) * inv_k)
    return a, den


def b_of_k(m: UnitMoments, k: float, cov: CovSpec) -> Estimate:
    a, den = family_terms(m, k)
    return ratio_estimate(a, den, m.d, cov, label=f"b(k={k})")


def split_iv(d: Draws, cov: CovSpec, min_valid: int = 8) -> Estimate:
    """Primary estimator: split-sample IV averaged over all splits, in closed form."""
    return b_of_k(unit_moments(d, min_valid), math.inf, cov)


def _first_valid(f: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.isfinite(f), axis=1)
    return f[np.arange(len(f)), idx]


def naive_single(d: Draws, cov: CovSpec, min_valid: int = 8) -> Estimate:
    """One draw per prompt (the first valid draw at each date): what the literature reports."""
    c = common_sample(d, min_valid)
    ft, ftm1 = _first_valid(c.f_t), _first_valid(c.f_tm1)
    et, rt = _demean(c.x - ft, c.cell), _demean(ft - ftm1, c.cell)
    return ratio_estimate(et * rt, rt**2, c, cov, label="naive_single")


def naive_stacked(d: Draws, cov: CovSpec, min_valid: int = 8) -> Estimate:
    """Every single draw, error and revision from the same draw index (same limit as one draw
    per prompt, less variance). Influence values are aggregated to the unit."""
    c = common_sample(d, min_valid)
    K = min(c.f_t.shape[1], c.f_tm1.shape[1])
    ft, ftm1 = c.f_t[:, :K], c.f_tm1[:, :K]
    ok = np.isfinite(ft) & np.isfinite(ftm1)
    rows_u = np.nonzero(ok)[0]
    e = (c.x[:, None] - ft)[ok]
    r = (ft - ftm1)[ok]
    cell = c.cell[rows_u]
    et, rt = _demean(e, cell), _demean(r, cell)
    a = np.bincount(rows_u, weights=et * rt, minlength=c.n)
    den = np.bincount(rows_u, weights=rt**2, minlength=c.n)
    return ratio_estimate(a, den, c, cov, label="naive_stacked")


def mean_of_k(
    d: Draws, k: int, cov: CovSpec, n_subsets: int = 50, seed: int = 0, min_valid: int = 8
) -> Estimate:
    """Regression on means of k randomly chosen valid draws per date; moments are averaged over
    `n_subsets` seeded subsets before taking the ratio."""
    c = common_sample(d, min_valid)
    rng = np.random.default_rng(seed)
    a_sum, den_sum = np.zeros(c.n), np.zeros(c.n)
    for _ in range(n_subsets):
        mt = _subset_mean(c.f_t, k, rng)
        mtm1 = _subset_mean(c.f_tm1, k, rng)
        et, rt = _demean(c.x - mt, c.cell), _demean(mt - mtm1, c.cell)
        a_sum += et * rt
        den_sum += rt**2
    return ratio_estimate(a_sum / n_subsets, den_sum / n_subsets, c, cov, label=f"mean_of_{k}")


def _subset_mean(f: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Mean of k valid draws chosen at random without replacement, per row."""
    keys = rng.random(f.shape)
    keys[~np.isfinite(f)] = np.inf
    order = np.argsort(keys, axis=1)[:, :k]
    picked = np.take_along_axis(f, order, axis=1)
    return picked.mean(1)


def split_iv_enumerated(d: Draws, min_valid: int = 2) -> float:
    """Test-only: average the split-sample IV numerator and denominator over every balanced
    split of each unit's draws (all draws valid, even counts), then take the ratio.

    Within a unit the average is computed by enumerating every split. Splits are independent
    across units, so the cross-unit terms of the cell-demeaned sums factor into per-unit
    averages; tests check that step separately by brute force on two units.
    """
    c = common_sample(d, min_valid)
    stats = np.array([_unit_split_moments(c.x[i], c.f_t[i], c.f_tm1[i]) for i in range(c.n)])
    Ezy, Ezr, Ez, Ey, Er = stats.T
    num = den = 0.0
    for cell in np.unique(c.cell):
        s = c.cell == cell
        n = s.sum()
        num += Ezy[s].sum() - (Ezy[s].sum() + Ez[s].sum() * Ey[s].sum() - (Ez[s] * Ey[s]).sum()) / n
        den += Ezr[s].sum() - (Ezr[s].sum() + Ez[s].sum() * Er[s].sum() - (Ez[s] * Er[s]).sum()) / n
    return num / den


def _halves(v: np.ndarray):
    K = len(v)
    assert K % 2 == 0, "enumeration needs an even number of draws"
    for a_idx in itertools.combinations(range(K), K // 2):
        mask = np.zeros(K, bool)
        mask[list(a_idx)] = True
        yield v[mask].mean(), v[~mask].mean()


def _unit_split_moments(x: float, ft: np.ndarray, ftm1: np.ndarray) -> tuple[float, ...]:
    """Averages over all (t-split, t-1-split) pairs of z*y, z*r, z, y, r, where
    y = x - Ft^A, r = Ft^A - Ftm1^A, z = Ft^B - Ftm1^B."""
    acc = np.zeros(5)
    n = 0
    for fa, fb in _halves(ft):
        for ga, gb in _halves(ftm1):
            y, r, z = x - fa, fa - ga, fb - gb
            acc += (z * y, z * r, z, y, r)
            n += 1
    return tuple(acc / n)


def split_iv_one_split(d: Draws, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Test-only: cell-demeaned IV numerator and denominator terms for one random balanced
    split of every unit (used to brute-force the cross-unit averaging)."""
    rows = []
    for i in range(d.n):
        pt = rng.permutation(d.f_t.shape[1])
        pm = rng.permutation(d.f_tm1.shape[1])
        h_t, h_m = len(pt) // 2, len(pm) // 2
        fa, fb = d.f_t[i, pt[:h_t]].mean(), d.f_t[i, pt[h_t:]].mean()
        ga, gb = d.f_tm1[i, pm[:h_m]].mean(), d.f_tm1[i, pm[h_m:]].mean()
        rows.append((d.x[i] - fa, fa - ga, fb - gb))
    y, r, z = np.array(rows).T
    zt = _demean(z, d.cell)
    return zt * _demean(y, d.cell), zt * _demean(r, d.cell)


def cross_template_iv(
    d: Draws, paraphrase: np.ndarray, path: np.ndarray, cov: CovSpec, min_valid: int = 8
) -> Estimate:
    """Secondary (crossed block): instrument each unit's revision with the mean revision of the
    units on the same path whose templates use a different paraphrase (plan D5)."""
    m = unit_moments(d, min_valid)
    keep = np.isin(d.uid, m.d.uid)
    par, pth = paraphrase[keep], path[keep]
    z = np.full(m.d.n, np.nan)
    for p in np.unique(pth):
        on = np.nonzero(pth == p)[0]
        for i in on:
            others = on[par[on] != par[i]]
            if len(others):
                z[i] = m.rbar[others].mean()
    ok = np.isfinite(z)
    sub = m.d.subset(ok)
    zt = _demean(z[ok], sub.cell)
    yt = _demean(m.ebar[ok], sub.cell)
    rt = _demean(m.rbar[ok], sub.cell)
    return ratio_estimate(zt * yt, zt * rt, sub, cov, label="cross_template_iv")


def difference(a: Estimate, b: Estimate, d: Draws, cov: CovSpec, label: str = "") -> Estimate:
    """a - b on the same units (H1, H5, LS contrasts), with the joint influence."""
    if not np.array_equal(a.uid, b.uid):
        raise ValueError("difference needs estimates on the same units")
    psi = a.psi - b.psi
    v, G = var_of_mean(psi, align(d, a.uid), cov)
    diff = a.b - b.b
    se = math.sqrt(v)
    return Estimate(
        b=diff,
        se=se,
        ci=(diff - Z95 * se, diff + Z95 * se),
        p=normal_p(diff / se),
        n_units=a.n_units,
        n_clusters=G,
        psi=psi,
        uid=a.uid,
        label=label or f"{a.label} - {b.label}",
    )


def align(d: Draws, uid: np.ndarray) -> Draws:
    """Rows of `d` in the order of `uid` (every uid must be present)."""
    pos = {u: i for i, u in enumerate(d.uid)}
    return d.subset(np.array([pos[u] for u in uid], dtype=int))


def by_cell(fn, d: Draws, cov: CovSpec, cells=None, **kw) -> dict:
    cells = np.unique(d.cell) if cells is None else cells
    return {c: fn(d.subset(d.cell == c), cov, **kw) for c in cells}


def pooled_equal_weight(by: dict, d: Draws, cov: CovSpec, cells=None, label="") -> Estimate:
    """Equal-weight mean of per-cell estimates (plan D5), with a pooled influence function so
    clusters shared across cells are handled."""
    cells = list(by) if cells is None else list(cells)
    ests = [by[c] for c in cells]
    uid = np.concatenate([e.uid for e in ests])
    U = len(uid)
    phi = np.concatenate([e.psi * U / (len(cells) * e.n_units) for e in ests])
    v, G = var_of_mean(phi, align(d, uid), cov)
    b = float(np.mean([e.b for e in ests]))
    se = math.sqrt(v)
    return Estimate(
        b=b,
        se=se,
        ci=(b - Z95 * se, b + Z95 * se),
        p=normal_p(b / se),
        n_units=U,
        n_clusters=G,
        psi=phi,
        uid=uid,
        label=label or "pooled_equal_weight",
    )
