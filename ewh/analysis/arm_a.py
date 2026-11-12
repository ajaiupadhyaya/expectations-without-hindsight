"""Arm A analysis: prompt table + raw draws -> every pre-registered estimate (plan D5, D10).

Shared by the CPU dry run (planted forecasters) and the confirmatory script. Inputs only: a
prompt table (ewh.prompts.tables) and a draws table (ewh.models.runner). Nothing here reads a
confirmatory file by itself; the confirm scripts pass the files in after their binding checks.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ewh.estimators import eor
from ewh.estimators.persistence import implied_persistence, noise_share
from ewh.estimators.types import CovSpec, Draws, Estimate
from ewh.prompts.parse import is_valid
from ewh.sim.benchmarks import ls_learner

# Primary pooled estimand (Decision 10): equal weights over rho in {0.4, 0.6, 0.8, 1}.
POOLED_RHOS = (0.4, 0.6, 0.8, 1.0)
RHO2_WEIGHTS = {0.2: 0.04, 0.4: 0.16, 0.6: 0.36, 0.8: 0.64, 1.0: 1.0}  # robustness
FAMILY = ("H1", "H2_LS", "H3")  # proposed Holm family (AJ decides in the PAP)
KS = (1, 2, 4, 8, 16, 32)
COV = CovSpec("cr1", ("path",))


def _codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s)[0]


def build_draws(table: pd.DataFrame, draws: pd.DataFrame, k: int, greedy: bool = False) -> Draws:
    """One Draws row per unit. Sampled draws (draw >= 0) are kept if valid under plan D6; with
    greedy=True the single greedy read (draw = -1) is used instead."""
    d = draws[draws.draw == -1] if greedy else draws[draws.draw >= 0]
    d = d[d.prompt_id.isin(set(table.prompt_id))]  # draws of other blocks are not this table's
    hist = {
        pid: np.array([float(v) for v in h.split(",")])
        for pid, h in zip(table.prompt_id, table["hist"], strict=True)
    }
    vals = d.parsed.to_numpy(float)
    ok = np.array(
        [
            is_valid(v if math.isfinite(v) else None, hist[p])
            for v, p in zip(vals, d.prompt_id, strict=True)
        ]
    )
    d = d.assign(value=np.where(ok, vals, np.nan))
    kk = 1 if greedy else k
    wide = d.pivot_table(index="prompt_id", columns="draw", values="value", dropna=False)
    wide = wide.reindex(columns=sorted(wide.columns)[:kk])
    t = table.set_index("prompt_id")
    units = table.drop_duplicates("unit_id").set_index("unit_id")
    ids = units.index.to_numpy()
    pid = {(u, dt): p for u, dt, p in zip(table.unit_id, table.date, table.prompt_id, strict=True)}

    def mat(date):
        rows = [pid[(u, date)] for u in ids]
        return wide.reindex(rows).to_numpy(float)

    return Draws(
        f_t=mat("t"),
        f_tm1=mat("tm1"),
        x=t.loc[[pid[(u, "t")] for u in ids], "target"].to_numpy(float),
        cell=units["rho"].to_numpy(float),
        clusters=np.column_stack(
            [
                _codes(units["path_uid"]),
                _codes(units["paraphrase"]),
                _codes(units["fmt"]),
                _codes(units["model"]),
            ]
        ),
        cluster_names=("path", "paraphrase", "fmt", "model"),
    )


def x_last_t(table: pd.DataFrame) -> np.ndarray:
    units = table.drop_duplicates("unit_id").unit_id
    s = table[table.date == "t"].set_index("unit_id").x_last
    return s.reindex(units).to_numpy(float)


def ls_benchmark(table: pd.DataFrame, k: int = 8) -> Draws:
    """The least-squares learner on exactly the histories shown, through the same estimators."""
    units = table.drop_duplicates("unit_id")
    h = {
        (u, dt): np.array([float(v) for v in hs.split(",")])
        for u, dt, hs in zip(table.unit_id, table.date, table["hist"], strict=True)
    }
    ids = units.unit_id.to_numpy()
    f_tm1 = np.array([ls_learner(h[(u, "tm1")][None, :], 2)[0] for u in ids])
    f_t = np.array([ls_learner(h[(u, "t")][None, :], 1)[0] for u in ids])
    tt = table[table.date == "t"].set_index("unit_id")
    return Draws(
        np.repeat(f_t[:, None], k, 1),
        np.repeat(f_tm1[:, None], k, 1),
        tt.loc[ids, "target"].to_numpy(float),
        units["rho"].to_numpy(float),
        np.column_stack([_codes(units["path_uid"]), _codes(units["paraphrase"])]),
        ("path", "paraphrase"),
    )


def slope_on_rho(by: dict[float, Estimate], subtract_rho: bool = True) -> dict:
    """H3: OLS slope of (implied - rho) on rho over independent cells."""
    rhos = np.array(sorted(by))
    y = np.array([by[r].b - (r if subtract_rho else 0) for r in rhos])
    se = np.array([by[r].se for r in rhos])
    w = (rhos - rhos.mean()) / ((rhos - rhos.mean()) ** 2).sum()
    b, s = float(w @ y), float(math.sqrt((w**2 * se**2).sum()))
    return dict(b=b, se=s, ci=(b - 1.96 * s, b + 1.96 * s), p=float(2 * (1 - _phi(abs(b / s)))))


def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _t(values) -> dict:
    """t-test across paraphrase-level estimates (template as a random factor; Decision 10)."""
    from scipy import stats

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    se = v.std(ddof=1) / math.sqrt(n)
    df = n - 1
    t = v.mean() / se if se > 0 else float("nan")
    crit = stats.t.ppf(0.975, df)
    return dict(
        b=float(v.mean()),
        se=float(se),
        ci=(v.mean() - crit * se, v.mean() + crit * se),
        p=float(2 * stats.t.sf(abs(t), df)),
        df=df,
        n_groups=n,
        mde=float((crit + stats.t.ppf(0.8, df)) * se),
        values=v.tolist(),
    )


def _paraphrase_stats(d: Draws, ls: Draws, xl: np.ndarray, cells) -> dict:
    """Per-paraphrase pooled IV, one-draw coefficient, LS benchmark and H3 slopes."""
    out = {k: [] for k in ("iv", "naive", "ls", "h3", "h3_ls")}
    par_d, par_l = d.column("paraphrase"), ls.column("paraphrase")
    for p in np.unique(par_d):
        dp, lp = d.subset(par_d == p), ls.subset(par_l == p)
        iv = eor.pooled_equal_weight(eor.by_cell(eor.split_iv, dp, COV), dp, COV, cells)
        nv = eor.pooled_equal_weight(eor.by_cell(eor.naive_stacked, dp, COV), dp, COV, cells)
        lsp = eor.pooled_equal_weight(eor.by_cell(eor.split_iv, lp, COV), lp, COV, cells)
        ip = eor.by_cell(lambda dd, cov: implied_persistence(dd, xl[dd.uid], cov), dp, COV)
        ipl = eor.by_cell(lambda dd, cov: implied_persistence(dd, xl[dd.uid], cov), lp, COV)
        out["iv"].append(iv.b)
        out["naive"].append(nv.b)
        out["ls"].append(lsp.b)
        out["h3"].append(slope_on_rho(ip)["b"])
        out["h3_ls"].append(slope_on_rho(ipl)["b"])
    return {k: np.array(v) for k, v in out.items()}


def analyse(table: pd.DataFrame, draws: pd.DataFrame, k: int) -> dict:
    """Every Arm A estimate. Primary inference: t-tests across the 12 paraphrases of
    per-paraphrase pooled statistics (equal weights over rho 0.4-1). Path-clustered versions over
    the full sample, rho^2 weights, the per-cell table and the noise ladder are robustness and
    descriptive rows."""
    d = build_draws(table, draws, k)
    g = build_draws(table, draws, k, greedy=True)
    xl = x_last_t(table)
    ls = ls_benchmark(table)
    ls = ls.subset(np.isin(ls.uid, eor.common_sample(d).uid))  # same units as the model
    out: dict = {
        "n_units": d.n,
        "noise_share_t": noise_share(d),
        "noise_share_tm1": noise_share(d, "tm1"),
    }
    est = {}
    for name, fn in (
        ("naive_single", eor.naive_single),
        ("naive_stacked", eor.naive_stacked),
        ("split_iv", eor.split_iv),
    ):
        est[name] = eor.by_cell(fn, d, COV)
    for kk in KS:
        if kk <= k:
            est[f"mean_of_{kk}"] = eor.by_cell(
                lambda dd, cov, kk=kk: eor.mean_of_k(dd, kk, cov, n_subsets=10), d, COV
            )
    est["greedy"] = eor.by_cell(eor.naive_single, g, COV, min_valid=1)
    est["ls_learner"] = eor.by_cell(eor.split_iv, ls, COV)
    out["by_rho"] = {name: {r: e for r, e in by.items()} for name, by in est.items()}

    cells = [r for r in POOLED_RHOS if r in est["split_iv"]]
    # primary: paraphrase t-tests
    ps = _paraphrase_stats(d, ls, xl, cells)
    out["H1"] = _t(ps["naive"] - ps["iv"])
    out["H2"] = _t(ps["iv"])
    out["H2_LS"] = _t(ps["iv"] - ps["ls"])
    out["H3"] = _t(ps["h3"])
    out["H3_LS"] = _t(ps["h3"] - ps["h3_ls"])
    from ewh.inference.multiple import holm, holm_adjusted

    pv = {h: out[h]["p"] for h in FAMILY}
    out["holm"] = dict(reject=holm(pv), adjusted=holm_adjusted(pv))

    # robustness: path-clustered over the full sample; rho^2 weights
    pooled = {
        name: eor.pooled_equal_weight(by, d if name != "ls_learner" else ls, COV, cells)
        for name, by in est.items()
        if name not in ("greedy",)
    }
    out["pooled"] = pooled
    r2 = {c: w for c, w in RHO2_WEIGHTS.items() if c in est["split_iv"]}
    out["rho2_weights"] = {
        name: eor.pooled_weighted(est[name], d, COV, r2) for name in ("split_iv", "naive_stacked")
    }
    out["path_clustered"] = {
        "H1": eor.difference(pooled["naive_stacked"], pooled["split_iv"], d, COV, "H1"),
        "H2": pooled["split_iv"],
        "H2_LS": eor.difference(pooled["split_iv"], pooled["ls_learner"], d, COV, "H2_LS"),
    }
    ip = eor.by_cell(lambda dd, cov: implied_persistence(dd, xl[dd.uid], cov), d, COV)
    ip_ls = eor.by_cell(lambda dd, cov: implied_persistence(dd, xl[dd.uid], cov), ls, COV)
    out["implied_persistence"] = ip
    out["path_clustered"]["H3"] = slope_on_rho(ip)
    out["path_clustered"]["H3_LS"] = _slope_contrast(ip, ip_ls)
    return out


def _slope_contrast(a: dict, b: dict) -> dict:
    """H3-LS (secondary): difference of slopes, conservative variance (sum of both)."""
    sa, sb = slope_on_rho(a), slope_on_rho(b)
    diff = sa["b"] - sb["b"]
    se = math.sqrt(sa["se"] ** 2 + sb["se"] ** 2)
    return dict(
        b=diff,
        se=se,
        ci=(diff - 1.96 * se, diff + 1.96 * se),
        p=float(2 * (1 - _phi(abs(diff / se)))),
    )


def summary_frame(res: dict) -> pd.DataFrame:
    rows = []
    for name, by in res["by_rho"].items():
        for r, e in by.items():
            rows.append(
                dict(
                    estimator=name,
                    rho=r,
                    b=e.b,
                    se=e.se,
                    lo=e.ci[0],
                    hi=e.ci[1],
                    n=e.n_units,
                    first_stage_t2=e.first_stage_t2,
                )
            )
    for name, e in res["pooled"].items():
        rows.append(
            dict(estimator=name, rho="pooled", b=e.b, se=e.se, lo=e.ci[0], hi=e.ci[1], n=e.n_units)
        )
    return pd.DataFrame(rows)
