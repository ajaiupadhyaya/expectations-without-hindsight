"""Competence gate (handoff 3.3, plan D9): checks unrelated to the hypotheses, run on gate seeds.

Input: the gate prompt table (ewh.prompts.tables.build_gate) for one model and an (n, K) array of
sampled forecasts aligned with its rows (NaN = unparsed). Thresholds are proposals until AJ fixes
them in the PAP; they are never changed after any model's gate result is seen.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ewh.prompts.parse import is_valid
from ewh.sim.benchmarks import ls_learner

THRESHOLDS = dict(
    parse_rate=0.95,
    history_slope=0.2,
    history_p=0.01,
    persistence_slope=0.25,
    scale_lo=0.8,
    scale_hi=1.2,
)


def _hist(table: pd.DataFrame) -> list[np.ndarray]:
    return [np.array([float(v) for v in h.split(",")]) for h in table["hist"]]


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Slope and its heteroskedasticity-robust (HC1) standard error."""
    xm, ym = x - x.mean(), y - y.mean()
    b = (xm @ ym) / (xm @ xm)
    u = ym - b * xm
    n = len(y)
    se = math.sqrt((xm**2 @ u**2) / (xm @ xm) ** 2 * n / (n - 2))
    return float(b), se


def gate_stats(table: pd.DataFrame, draws: np.ndarray, th: dict = THRESHOLDS) -> dict:
    hists = _hist(table)
    valid = np.array(
        [
            [is_valid(v if np.isfinite(v) else None, h) for v in row]
            for row, h in zip(draws, hists, strict=True)
        ]
    )
    parse_rate = float(valid.mean())
    f = np.where(valid, draws, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows with no valid draw -> NaN
        fmean = np.nanmean(f, axis=1)
    kind = table["kind"].to_numpy()
    xl = table["x_last"].to_numpy(float)

    # persistence: implied persistence by rho, then its slope across rho (independent cells)
    per = {}
    for rho in (0.0, 0.5, 1.0):
        m = (kind == "persistence") & np.isclose(table["rho"].to_numpy(float), rho)
        m &= np.isfinite(fmean)
        per[rho] = _ols(fmean[m], xl[m]) if m.sum() > 2 else (np.nan, np.inf)
    slope = per[1.0][0] - per[0.0][0]
    slope_se = math.sqrt(per[1.0][1] ** 2 + per[0.0][1] ** 2)
    persist_lo = slope - 1.959964 * slope_se

    # history sensitivity: same last value, different histories
    a = table[kind == "hist_a"].reset_index()
    b = table[kind == "hist_b"].reset_index()
    ia, ib = a["index"].to_numpy(), b["index"].to_numpy()
    ls_a = ls_learner(np.stack([hists[i] for i in ia]), 1)
    ls_b = ls_learner(np.stack([hists[i] for i in ib]), 1)
    dy, dx = fmean[ib] - fmean[ia], ls_b - ls_a
    ok = np.isfinite(dy)
    h_slope, h_se = _ols(dy[ok], dx[ok]) if ok.sum() > 2 else (np.nan, np.inf)
    h_p = float(stats.norm.sf(h_slope / h_se)) if np.isfinite(h_slope) and h_se > 0 else 1.0

    # scale invariance: forecasts of 3x + 500 against 3 F(x) + 500
    o = table.index[kind == "scale_orig"].to_numpy()
    s = table.index[kind == "scale_trans"].to_numpy()
    y, x = fmean[s], 3 * fmean[o] + 500
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() > 2 and np.var(x[ok]) > 0:
        sc_slope, _ = _ols(y[ok], x[ok])
    else:
        sc_slope = np.nan

    res = dict(
        parse_rate=parse_rate,
        implied_rho={k: v[0] for k, v in per.items()},
        persistence_slope=slope,
        persistence_lo=persist_lo,
        history_slope=h_slope,
        history_p=h_p,
        scale_slope=sc_slope,
    )
    res["pass_parse"] = parse_rate >= th["parse_rate"]
    res["pass_persistence"] = bool(slope >= th["persistence_slope"] and persist_lo > 0)
    res["pass_history"] = bool(h_slope >= th["history_slope"] and h_p < th["history_p"])
    res["pass_scale"] = bool(th["scale_lo"] <= sc_slope <= th["scale_hi"])
    res["pass"] = all(
        res[k] for k in ("pass_parse", "pass_persistence", "pass_history", "pass_scale")
    )
    return res


# ---------------------------------------------------------------- synthetic forecasters
def synthetic(table: pd.DataFrame, kind: str, k: int, rng: np.random.Generator) -> np.ndarray:
    """(n, k) draws from benchmark forecasters used to validate the gate before the freeze."""
    hists = np.stack(_hist(table))
    last, mean = hists[:, -1], hists.mean(1)
    if kind == "ls":
        f = ls_learner(hists, 1)
        noise = 2.0
    elif kind == "overreactor":  # AKLMT's pattern: implied persistence 0.45 + 0.54 * LS estimate
        xs, ys = hists[:, :-1], hists[:, 1:]
        xm, ym = xs - xs.mean(1, keepdims=True), ys - ys.mean(1, keepdims=True)
        b = (xm * ym).sum(1) / (xm**2).sum(1)
        f = mean + (0.45 + 0.54 * b) * (last - mean)
        noise = 2.0
    elif kind == "echo":
        f, noise = last, 0.0
    elif kind == "noisy_echo":
        f, noise = last, 0.25 * hists.std(1, ddof=1)
    elif kind == "constant":
        f, noise = np.zeros(len(last)), 0.0
    else:
        raise ValueError(kind)
    scale = np.maximum(noise, 0.0)
    return f[:, None] + rng.normal(0, 1, (len(f), k)) * np.atleast_1d(scale)[:, None]
