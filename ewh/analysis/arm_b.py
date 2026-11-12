"""Arm B analysis: real-time macro forecasts from vintage models (plan D11, D12, Decision 7, 10).

Inputs: the Arm B prompt table (ewh.prompts.tables.build_arm_b), the draws of the models that ran
it, and an assignment of a model to each pair (ewh.data.pairs). A unit is one pair (variable, t)
under one framing and one template. Errors, revisions and outcomes are divided by each
variable's pre-2000 standard deviation, and every regression has variable intercepts.

Primary pooled inference is block-t over 8 consecutive time blocks (7 df; docs/power.md); the
paraphrase t-test (11 df) is a robustness row. H5 and the leakage secondaries are contrasts
estimated block by block on the same pairs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from ewh.data import pairs
from ewh.data import realtime as rt
from ewh.estimators import eor
from ewh.estimators.types import CovSpec, Draws
from ewh.prompts.parse import is_valid

BLOCKS = 8
COVID = (pd.Period("2020Q1"), pd.Period("2021Q2"))
COV = CovSpec("cr1", ("quarter",))


def pre2000_sd(var: str) -> float:
    """S.d. of the outcome over target quarters 1985Q1-1999Q4, from the vintage dated 2000Q1."""
    s = rt.transform(var, rt.series_in_vintage(var, pd.Period("2000Q1")))
    s = s[(s.index >= pd.Period("1985Q1")) & (s.index <= pd.Period("1999Q4"))]
    return float(s.std(ddof=1))


def assignment(rule: str, vintage: int | None = None, passing=None, six: bool = False):
    """Map (variable, t, date) -> model name `chrono-YYYY` under a pair rule."""

    def f(var: str, t: pd.Period, date: str) -> str | None:
        if rule == "fixed":
            return f"chrono-{vintage}"
        v_tm1, v_t = pairs.assign(t, rule)
        if six:
            v_tm1, v_t = pairs.six_vintage_fallback(v_tm1), pairs.six_vintage_fallback(v_t)
        if passing is not None:
            v_tm1, v_t = pairs.substitute(v_tm1, passing), pairs.substitute(v_t, passing)
        v = v_tm1 if date == "tm1" else v_t
        return None if v is None else f"chrono-{v}"

    return f


def build_draws_b(
    table: pd.DataFrame,
    draws: pd.DataFrame,
    assign: Callable,
    k: int,
    framing: str = "named",
    covid: bool = True,
    pair_filter: Callable | None = None,
) -> tuple[Draws, pd.DataFrame]:
    """Draws for one run. Returns the Draws and a unit frame (variable, t, template, ...)."""
    tb = table[table.framing == framing].copy()
    tb["t_p"] = tb.t.map(pd.Period)
    if covid:
        tgt = tb.t_p + 3
        tb = tb[~((tgt >= COVID[0]) & (tgt <= COVID[1]))]
    if pair_filter is not None:
        tb = tb[[pair_filter(v, t) for v, t in zip(tb.variable, tb.t_p, strict=True)]]
    tb["model"] = [assign(v, t, d) for v, t, d in zip(tb.variable, tb.t_p, tb.date, strict=True)]
    tb = tb[tb.model.notna()]
    d = draws[draws.draw >= 0][["model", "prompt_id", "draw", "parsed"]]
    m = tb[["prompt_id", "model", "hist"]].merge(d, on=["model", "prompt_id"], how="left")
    hist = {
        pid: np.array([float(v) for v in h.split(",")])
        for pid, h in zip(tb.prompt_id, tb["hist"], strict=True)
    }
    ok = [
        is_valid(v if (v is not None and math.isfinite(v)) else None, hist[p])
        for v, p in zip(m.parsed, m.prompt_id, strict=True)
    ]
    m["value"] = np.where(ok, m.parsed, np.nan)
    wide = m.pivot_table(index=["model", "prompt_id"], columns="draw", values="value", dropna=False)
    wide = wide.reindex(columns=range(k))
    units = tb.drop_duplicates("unit_id").set_index("unit_id")
    key = {
        (u, dt): (mo, p)
        for u, dt, mo, p in zip(tb.unit_id, tb.date, tb.model, tb.prompt_id, strict=True)
    }
    ids = [u for u in units.index if (u, "tm1") in key and (u, "t") in key]
    units = units.loc[ids]
    sd = {v: pre2000_sd(v) for v in units.variable.unique()}
    scale = units.variable.map(sd).to_numpy(float)

    def mat(date):
        rows = [key[(u, date)] for u in ids]
        return wide.reindex(rows).to_numpy(float) / scale[:, None]

    quarter = units.t.map(lambda s: pd.Period(s).ordinal).to_numpy()
    out = Draws(
        f_t=mat("t"),
        f_tm1=mat("tm1"),
        x=units.target.to_numpy(float) / scale,
        cell=pd.factorize(units.variable)[0].astype(float),
        clusters=np.column_stack(
            [quarter, pd.factorize(units.paraphrase)[0], pd.factorize(units.template)[0]]
        ),
        cluster_names=("quarter", "paraphrase", "template"),
    )
    info = units.reset_index()[["unit_id", "variable", "t", "paraphrase", "fmt", "template"]]
    info["quarter"] = quarter
    return out, info


def blocks(quarters: np.ndarray, n: int = BLOCKS) -> np.ndarray:
    """Block id (0..n-1) of each unit: consecutive, nearly equal groups of distinct quarters."""
    qs = np.sort(np.unique(quarters))
    edges = np.array_split(qs, n)
    lookup = {q: i for i, blk in enumerate(edges) for q in blk}
    return np.array([lookup[q] for q in quarters])


def pooled_b(d: Draws, estimator=eor.split_iv) -> float:
    """Pooled coefficient with variable intercepts (cell = variable)."""
    return estimator(d, COV).b


def t_test(values: np.ndarray) -> dict:
    """t-test across groups; groups with a non-finite estimate are counted and reported, not
    silently dropped (n_nonfinite)."""
    v = np.asarray(values, float)
    n_bad = int((~np.isfinite(v)).sum())
    v = v[np.isfinite(v)]
    n = len(v)
    se = v.std(ddof=1) / math.sqrt(n)
    t = v.mean() / se if se > 0 else float("nan")
    crit = stats.t.ppf(0.975, n - 1)
    return dict(
        b=float(v.mean()),
        se=float(se),
        ci=(v.mean() - crit * se, v.mean() + crit * se),
        p=float(2 * stats.t.sf(abs(t), n - 1)),
        df=n - 1,
        n_groups=n,
        n_nonfinite=n_bad,
    )


def block_estimates(d: Draws, info: pd.DataFrame, estimator=eor.split_iv) -> np.ndarray:
    blk = blocks(info.quarter.to_numpy())
    return np.array([pooled_b(d.subset(blk == i), estimator) for i in range(BLOCKS)])


def block_t(d: Draws, info: pd.DataFrame, estimator=eor.split_iv) -> dict:
    """Primary pooled inference (D12): mean of the block coefficients, t-test with 7 df; the
    point estimate reported beside it is the full-sample pooled coefficient."""
    res = t_test(block_estimates(d, info, estimator))
    res["b_full"] = pooled_b(d, estimator)
    return res


def paraphrase_t(d: Draws, info: pd.DataFrame, estimator=eor.split_iv) -> dict:
    """Robustness: per-paraphrase pooled coefficient, t-test across paraphrases."""
    par = info.paraphrase.to_numpy()
    return t_test([pooled_b(d.subset(par == p), estimator) for p in np.unique(par)])


def paired_contrast(a: tuple[Draws, pd.DataFrame], b: tuple[Draws, pd.DataFrame]) -> dict:
    """H5-type contrast on the same pairs: block-by-block difference of pooled coefficients."""
    da, ia = a
    db, ib = b
    common = sorted(set(ia.unit_id) & set(ib.unit_id))
    sa = ia.unit_id.isin(common).to_numpy()
    sb = ib.unit_id.isin(common).to_numpy()
    da, ia = da.subset(sa), ia[sa].reset_index(drop=True)
    db, ib = db.subset(sb), ib[sb].reset_index(drop=True)
    oa, ob = np.argsort(ia.unit_id.to_numpy()), np.argsort(ib.unit_id.to_numpy())
    da, ia = da.subset(oa), ia.iloc[oa].reset_index(drop=True)
    db, ib = db.subset(ob), ib.iloc[ob].reset_index(drop=True)
    blk = blocks(ia.quarter.to_numpy())
    diffs = [pooled_b(da.subset(blk == i)) - pooled_b(db.subset(blk == i)) for i in range(BLOCKS)]
    res = t_test(diffs)
    res["b_full"] = pooled_b(da) - pooled_b(db)
    res["n_units"] = len(common)
    return res


def per_variable(d: Draws, info: pd.DataFrame, lags: int = 4) -> pd.DataFrame:
    """Per-variable split IV with Newey-West errors over quarters (Bartlett, `lags`)."""
    rows = []
    for v in info.variable.unique():
        s = (info.variable == v).to_numpy()
        e = eor.split_iv(d.subset(s), CovSpec("nw", time="quarter", lags=lags))
        rows.append(
            dict(
                variable=v,
                b=e.b,
                se=e.se,
                lo=e.ci[0],
                hi=e.ci[1],
                n=e.n_units,
                first_stage_t2=e.first_stage_t2,
            )
        )
    return pd.DataFrame(rows)


def rmse(d: Draws) -> float:
    """Root mean squared error of the mean valid forecast at date t (scaled units)."""
    f = np.nanmean(d.f_t, 1)
    e = d.x - f
    return float(np.sqrt(np.nanmean(e**2)))
