"""Arm B revision pairs: which ChronoGPT vintage produces each forecast (plan D11, Decision 7).

A pair is (variable, forecast quarter t): the forecast made at survey quarter t-1 and the one made
at t, of the same target. ChronoGPT vintage Y has training cutoff 31 December Y; a forecast at
survey quarter s may use a vintage only if Y <= year(s) - 1.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ewh.data import realtime as rt

VINTAGES = tuple(range(1999, 2025))
PRIORITY = (1999, 2004, 2009, 2014, 2019, 2024)
FIRST_T = pd.Period("2000Q2")


def valid_vintage(s: pd.Period) -> int:
    """Latest vintage allowed at survey quarter s."""
    return s.year - 1


def assign(t: pd.Period, rule: str = "primary") -> tuple[int, int]:
    """(vintage for the t-1 forecast, vintage for the t forecast).

    primary: both use the vintage valid at t-1, so a revision never mixes two models.
    latest: each forecast uses the latest vintage valid at its own date (robustness).
    """
    if rule == "primary":
        v = valid_vintage(t - 1)
        return v, v
    if rule == "latest":
        return valid_vintage(t - 1), valid_vintage(t)
    raise ValueError(rule)


def substitute(v: int, passing: Iterable[int]) -> int | None:
    """Replace a vintage that failed the gate by the latest passing vintage that is no later
    (still leakage-free). None if no earlier vintage passed."""
    ok = [p for p in passing if p <= v]
    return max(ok) if ok else None


def six_vintage_fallback(v: int) -> int:
    """Pre-registered fallback when not all 26 vintages complete: latest priority vintage <= v."""
    return max(p for p in PRIORITY if p <= v)


def vintage_date(q: pd.Period) -> pd.Timestamp:
    """Publication date of an RTDSM quarterly vintage: the 15th of the quarter's middle month."""
    return q.start_time + pd.DateOffset(months=1, days=14)


def contaminated(t: pd.Period, cutoff_year: int = 2024) -> bool:
    """True if the pair's target was published (vintage t+4) on or before 31 Dec cutoff_year,
    so a model with that cutoff may have seen the outcome."""
    return vintage_date(t + 4) <= pd.Timestamp(f"{cutoff_year}-12-31")


def forecast_quarters(var: str) -> pd.PeriodIndex:
    return pd.period_range(FIRST_T, rt.last_forecast_quarter(var), freq="Q")


def pair_table(
    variables: Iterable[str] = rt.VARIABLES, rule: str = "primary", passing=None, six=False
) -> pd.DataFrame:
    rows = []
    for var in variables:
        for t in forecast_quarters(var):
            v_tm1, v_t = assign(t, rule)
            if six:
                v_tm1, v_t = six_vintage_fallback(v_tm1), six_vintage_fallback(v_t)
            if passing is not None:
                v_tm1, v_t = substitute(v_tm1, passing), substitute(v_t, passing)
            rows.append(
                dict(
                    variable=var,
                    t=t,
                    target_quarter=t + 3,
                    vintage_tm1=v_tm1,
                    vintage_t=v_t,
                    contaminated_2024=contaminated(t),
                )
            )
    return pd.DataFrame(rows)
