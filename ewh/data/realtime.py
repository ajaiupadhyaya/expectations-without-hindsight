"""Real-time data: RTDSM vintage matrices, real-time histories and first-vintage outcomes.

A vintage matrix has one row per observation quarter and one column per vintage quarter
(pandas Periods). Vintage s holds what was published by the middle of quarter s, so its last
observation is normally s-1. Rules are in docs/decisions.md, Decision 7.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from ewh.data.philfed import SNAPSHOT
from ewh.provenance.archive import RAW_ROOT

GROWTH = (
    "NGDP", "RGDP", "PGDP", "RCONSUM", "RNRESIN", "RRESINV", "RGF", "RGSL", "HOUSING", "CPI",
    "INDPROD",
)  # fmt: skip
LEVEL = ("UNEMP", "TB3M", "TN10Y")
VARIABLES = GROWTH + LEVEL

# variable -> (RTDSM file stem, layout): "Q" quarterly obs and vintages; "QM" quarterly
# vintages of monthly obs; "MM" monthly vintages of monthly obs.
RTDSM = {
    "NGDP": ("NOUTPUTQvQd", "Q"),
    "RGDP": ("ROUTPUTQvQd", "Q"),
    "PGDP": ("PQvQd", "Q"),
    "RCONSUM": ("RCONQvQd", "Q"),
    "RNRESIN": ("rinvbfQvQd", "Q"),
    "RRESINV": ("rinvresidQvQd", "Q"),
    "RGF": ("RGFQvQd", "Q"),
    "RGSL": ("RGSLQvQd", "Q"),
    "UNEMP": ("rucQvMd", "QM"),
    "CPI": ("cpiQvMd", "QM"),
    "INDPROD": ("iptMvMd", "MM"),
    "HOUSING": ("hstartsMvMd", "MM"),
}
FRED = {"TB3M": "TB3MS", "TN10Y": "GS10"}
# Monthly-vintage files: the month standing for each quarterly vintage. RTDSM quarterly vintages
# are dated in the middle month of the quarter, near the SPF deadline, so Arm B uses month 2.
# BGMS used month 1 for INDPROD (their 1.realtime_clean.do); the replication passes 1.
MONTH_IN_QUARTER = 2

_COL = re.compile(r"^[A-Za-z]+(\d{2})(Q|M)(\d{1,2})$")


def _year(yy: str) -> int:
    y = int(yy)
    return 1900 + y if y >= 60 else 2000 + y


def _vintage(col: str) -> pd.Period | None:
    m = _COL.match(col)
    if not m:
        return None
    y, kind, n = _year(m.group(1)), m.group(2), int(m.group(3))
    return pd.Period(
        f"{y}Q{n}" if kind == "Q" else f"{y}-{n:02d}", freq="Q" if kind == "Q" else "M"
    )


def _obs_index(dates: pd.Series) -> pd.PeriodIndex:
    s = dates.astype(str).str.replace(":", "-", regex=False)
    if s.str.contains("Q").all():
        return pd.PeriodIndex(s.str.replace("-", "", regex=False), freq="Q")
    return pd.PeriodIndex(s, freq="M")


def _monthly_to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly means of monthly rows; NaN unless all three months are present."""
    q = df.index.asfreq("Q")
    g = df.groupby(q)
    return g.mean().where(g.count() == 3)


@cache
def vintage_matrix(var: str, month_in_quarter: int = MONTH_IN_QUARTER, root: Path = RAW_ROOT):
    """Observation quarter x vintage quarter matrix for an RTDSM variable."""
    stem, layout = RTDSM[var]
    path = root / "rtdsm" / SNAPSHOT / f"{stem}.xlsx"
    raw = pd.read_excel(path, engine="calamine")
    raw.index = _obs_index(raw.pop("DATE"))
    raw = raw.apply(pd.to_numeric, errors="coerce")
    vint = {c: _vintage(c) for c in raw.columns}
    raw = raw[[c for c in raw.columns if vint[c] is not None]]
    if layout == "MM":
        keep = {
            c: p
            for c in raw.columns
            if (p := vint[c]) is not None and (p.month - 1) % 3 == month_in_quarter - 1
        }
        raw = raw[list(keep)].rename(columns={c: p.asfreq("Q") for c, p in keep.items()})
    else:
        raw = raw.rename(columns=vint)
    if layout in ("QM", "MM"):
        raw = _monthly_to_quarterly(raw)
    raw.columns = pd.PeriodIndex(raw.columns, freq="Q")
    return raw.sort_index().sort_index(axis=1)


@cache
def fred_quarterly(var: str, root: Path = RAW_ROOT) -> pd.Series:
    path = root / "fred" / SNAPSHOT / f"{FRED[var]}.csv"
    df = pd.read_csv(path)
    s = pd.Series(
        pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
        index=pd.PeriodIndex(pd.to_datetime(df.iloc[:, 0]), freq="M"),
    )
    q = s.groupby(s.index.asfreq("Q"))
    return q.mean().where(q.count() == 3).dropna()


def first_available_vintage(m: pd.DataFrame, s: pd.Period) -> pd.Period:
    """Vintage s, or the next vintage with data if s is missing (e.g. 1996Q1, government
    shutdown)."""
    for v in m.columns[m.columns >= s]:
        if m[v].notna().any():
            return v
    raise KeyError(f"no vintage at or after {s}")


def value(var: str, obs: pd.Period, vintage: pd.Period, **kw) -> float:
    """x_obs as published in `vintage` (rates: never revised, known if obs < vintage)."""
    if var in FRED:
        s = fred_quarterly(var)
        return float(s[obs]) if obs < vintage and obs in s.index else np.nan
    m = vintage_matrix(var, **kw)
    v = first_available_vintage(m, vintage)
    return float(m.at[obs, v]) if obs in m.index else np.nan


def series_in_vintage(var: str, vintage: pd.Period, **kw) -> pd.Series:
    """Every observation published in `vintage`, ending at the last available quarter."""
    if var in FRED:
        s = fred_quarterly(var)
        return s[s.index < vintage]
    m = vintage_matrix(var, **kw)
    return m[first_available_vintage(m, vintage)].dropna()


def transform(var: str, levels: pd.Series) -> pd.Series:
    """Four-quarter growth in percent for growth variables, levels otherwise."""
    if var in GROWTH:
        return (100 * (levels / levels.shift(4) - 1)).dropna()
    return levels


def history(var: str, s: pd.Period, n: int = 40, **kw) -> pd.Series:
    """The last n transformed values available at survey quarter s, from vintage s only.
    Raises if the vintage does not end at s-1 (the information set would be stale)."""
    lv = series_in_vintage(var, s, **kw)
    full = lv.reindex(pd.period_range(lv.index.min(), lv.index.max(), freq="Q"))
    tr = transform(var, full).dropna()
    if tr.index.max() != s - 1:
        raise ValueError(f"{var} vintage {s} ends at {tr.index.max()}, expected {s - 1}")
    return tr.iloc[-n:]


def outcome(var: str, t: pd.Period, **kw) -> float:
    """Target realized: growth x_{t+3}/x_{t-1} - 1 (percent) or level x_{t+3}, both from the
    first vintage containing x_{t+3} (dated t+4)."""
    if var in FRED:
        s = fred_quarterly(var)
        return float(s.get(t + 3, np.nan))
    v = t + 4
    num = value(var, t + 3, v, **kw)
    if var in LEVEL:
        return num
    return 100 * (num / value(var, t - 1, v, **kw) - 1)


def last_forecast_quarter(var: str, **kw) -> pd.Period:
    """Last t whose outcome is published in the snapshot (vintage t+4 contains x_{t+3})."""
    if var in FRED:
        return fred_quarterly(var).index.max() - 3
    m = vintage_matrix(var, **kw)
    t = m.columns.max() - 4
    while not np.isfinite(outcome(var, t, **kw)):
        t -= 1
    return t
