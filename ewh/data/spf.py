"""SPF individual forecasts as revision pairs, following BGMS (2020) exactly (Decision 7).

`spf_pairs(var, mode)` returns one row per (forecaster, forecast quarter t) with the forecast of
the t+3 target made at t, the t-1 survey's forecast of the same target, the revision, the
outcome and the error, in percent (growth) or levels. mode="bgms" reproduces BGMS's data
choices for the replication check; mode="armb" uses the pure vintage rule of Decision 7.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from ewh.data import realtime as rt
from ewh.data.philfed import SNAPSHOT
from ewh.provenance.archive import RAW_ROOT

SHEET = {
    "NGDP": "NGDP", "RGDP": "RGDP", "PGDP": "PGDP", "CPI": "CPI", "INDPROD": "INDPROD",
    "RCONSUM": "RCONSUM", "RNRESIN": "RNRESIN", "RRESINV": "RRESINV", "RGF": "RFEDGOV",
    "RGSL": "RSLGOV", "HOUSING": "HOUSING", "UNEMP": "UNEMP", "TB3M": "TBILL", "TN10Y": "TBOND",
}  # fmt: skip
TRIM_IQR = 5.0
MIN_PAIRS = 10


@cache
def load(var: str) -> pd.DataFrame:
    """Raw individual forecasts: columns yq (Period), ID, c1..c6 (quarters t-1 .. t+4)."""
    path = RAW_ROOT / "spf" / SNAPSHOT / "SPFmicrodata.xlsx"
    sheet = SHEET[var]
    df = pd.read_excel(path, sheet_name=sheet, engine="calamine")
    out = pd.DataFrame(
        {
            "yq": pd.PeriodIndex(
                [pd.Period(f"{int(y)}Q{int(q)}") for y, q in zip(df.YEAR, df.QUARTER, strict=True)]
            ),
            "ID": df.ID.astype(int),
        }
    )
    for k in range(1, 7):
        out[f"c{k}"] = pd.to_numeric(df[f"{sheet}{k}"], errors="coerce")
    return out


def stata_pctile(x: np.ndarray, p: float) -> float:
    """Stata's default (_pctile) percentile, which BGMS's `egen pctile` uses."""
    x = np.sort(x[np.isfinite(x)])
    n = len(x)
    if n == 0:
        return np.nan
    npk = n * p / 100
    k = int(np.floor(npk))
    if np.isclose(npk, k):
        return float(x[max(k - 1, 0)] + x[min(k, n - 1)]) / 2
    return float(x[min(k, n - 1)])


def trim(df: pd.DataFrame, c: float = TRIM_IQR) -> pd.DataFrame:
    """Set to missing any forecast in columns 2..6 more than c IQRs from the cross-sectional
    median of its survey quarter (BGMS 2.spf_clean.do)."""
    out = df.copy()
    for k in range(2, 7):
        col = f"c{k}"
        for _, idx in out.groupby("yq").groups.items():
            v = out.loc[idx, col].to_numpy(float)
            med, p25, p75 = (stata_pctile(v, p) for p in (50, 25, 75))
            bad = (v < med - c * (p75 - p25)) | (v > med + c * (p75 - p25))
            out.loc[np.asarray(idx)[bad], col] = np.nan
    return out


def _kw(var: str, mode: str) -> dict:
    if mode == "bgms" and var == "INDPROD":
        return {"month_in_quarter": 1, "obs_agg": "last"}
    return {}


@cache
def _housing_releases() -> tuple[pd.Series, pd.Series]:
    """BGMS housing actuals: (first available release, most recent release), quarterly means,
    in millions (SPF units)."""
    path = RAW_ROOT / "rtdsm" / SNAPSHOT / "hstarts_first_second_third.xlsx"
    d = pd.read_excel(path, sheet_name="DATA", engine="calamine", header=None)
    d = d[d[0].astype(str).str.fullmatch(r"\d{4}:\d{2}")]
    idx = pd.PeriodIndex(d[0].astype(str).str.replace(":", "-"), freq="M")
    vals = d[[1, 2, 3, 4]].apply(pd.to_numeric, errors="coerce")
    vals.index = idx
    first = vals[1].fillna(vals[2]).fillna(vals[3]).fillna(vals[4])
    q = idx.asfreq("Q")
    return first.groupby(q).mean() / 1000, vals[4].groupby(q).mean() / 1000


def outcome(var: str, t: pd.Period, mode: str) -> float:
    if mode == "bgms" and var == "HOUSING":
        _, most = _housing_releases()
        return 100 * (most.get(t + 3, np.nan) / most.get(t - 1, np.nan) - 1)
    return rt.outcome(var, t, **_kw(var, mode))


def base_tm1(var: str, t: pd.Period, mode: str) -> float:
    """Real-time x_{t-1} known at survey t, the denominator of the SPF growth forecast."""
    if mode == "bgms" and var == "HOUSING":
        first, _ = _housing_releases()
        return float(first.get(t - 1, np.nan))
    kw = _kw(var, mode)
    if mode == "bgms" and var in rt.RTDSM and rt.RTDSM[var][1] == "Q":
        m = rt.vintage_matrix(var, **kw)
        if t in m.columns and np.isnan(m.at[t - 1, t]):
            if var == "NGDP":
                return _bgms_spline_base(var, t, kw)
            return _bgms_delayed_base(var, t, kw)
    return rt.value(var, t - 1, t, **kw)


def _bgms_spline_base(var: str, t: pd.Period, kw: dict) -> float:
    """BGMS's NGDP fix for 1996Q1 (Stata `csipolate` over years <= 1996): natural cubic spline
    through the real-time base series, evaluated at t, with no base rescaling."""
    from scipy.interpolate import CubicSpline

    qs = [q for q in pd.period_range("1968Q4", "1996Q4", freq="Q") if q != t]
    pts = [(q.ordinal, rt.value(var, q - 1, q, **kw)) for q in qs]
    pts = [(x, y) for x, y in pts if np.isfinite(y)]
    xs, ys = zip(*pts, strict=True)
    return float(CubicSpline(xs, ys, bc_type="natural")(t.ordinal))


def _bgms_delayed_base(var: str, t: pd.Period, kw: dict) -> float:
    """BGMS's fix when vintage t lacks x_{t-1} (1996Q1): extrapolate the real-time base linearly
    from the two previous surveys (Stata `ipolate, epolate`) and rescale by the average ratio of
    vintage t to vintage t-1 over the eight quarters before t-1 (their `nvo`), so the base is in
    the units of the forecasts made at t."""
    b1 = rt.value(var, t - 2, t - 1, **kw)
    b2 = rt.value(var, t - 3, t - 2, **kw)
    m = rt.vintage_matrix(var, **kw)
    ratios = [m.at[t - 1 - k, t] / m.at[t - 1 - k, t - 1] for k in range(1, 9)]
    return (2 * b1 - b2) * float(np.mean(ratios))


def spf_pairs(var: str, mode: str = "bgms") -> pd.DataFrame:
    df = trim(load(var))
    cur = df.set_index(["ID", "yq"])
    prev = df.assign(yq=df.yq + 1).set_index(["ID", "yq"])  # the same forecaster's t-1 survey
    j = cur.join(prev, suffixes=("", "_p"), how="left").reset_index()
    quarters = j.yq.unique()

    if var == "CPI":
        g = lambda frame, ks: np.prod([1 + frame[k] / 400 for k in ks], axis=0) - 1  # noqa: E731
        fc = g(j, ["c2", "c3", "c4", "c5"])
        fc_prev = g(j, ["c3_p", "c4_p", "c5_p", "c6_p"])
        fc, fc_prev = 100 * fc, 100 * fc_prev
    elif var in rt.LEVEL:
        fc, fc_prev = j.c5, j.c6_p
    else:
        if var == "INDPROD":
            base = j.c1
        else:
            bases = {q: base_tm1(var, q, mode) for q in quarters}
            base = j.yq.map(bases)
        fc = 100 * (j.c5 / base - 1)
        fc_prev = 100 * (j.c6_p / j.c2_p - 1)

    outs = {q: outcome(var, q, mode) for q in quarters}
    res = pd.DataFrame(
        {"ID": j.ID, "t": j.yq, "fc": fc, "fc_prev": fc_prev, "outcome": j.yq.map(outs)}
    )
    res["rev"] = res.fc - res.fc_prev
    res["err"] = res.outcome - res.fc
    return res


def sample(pairs: pd.DataFrame, first: pd.Period, last: pd.Period) -> pd.DataFrame:
    """Restrict to quarters [first, last] and to forecasters with at least MIN_PAIRS complete
    pairs there (BGMS keep forecasters with >= 10)."""
    s = pairs[(pairs.t >= first) & (pairs.t <= last)].copy()
    complete = s.err.notna() & s.rev.notna()
    counts = complete.groupby(s.ID).transform("sum")
    s.loc[counts < MIN_PAIRS, ["err", "rev"]] = np.nan
    return s


def consensus(s: pd.DataFrame) -> pd.DataFrame:
    """Mean of individual errors and revisions by quarter (BGMS baseline.do)."""
    return s.groupby("t")[["err", "rev"]].mean().reset_index()
