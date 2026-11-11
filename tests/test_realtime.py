import numpy as np
import pandas as pd
import pytest

from ewh.data import realtime as rt
from ewh.data.philfed import SNAPSHOT
from ewh.provenance.archive import RAW_ROOT

P = pd.Period
pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "rtdsm" / SNAPSHOT / "ROUTPUTQvQd.xlsx").exists(),
    reason="RTDSM snapshot not archived (python -m ewh.data.philfed)",
)


@pytest.fixture(scope="module")
def raw_routput():
    """The RGDP vintage file read without the loader, as an independent check."""
    df = pd.read_excel(RAW_ROOT / "rtdsm" / SNAPSHOT / "ROUTPUTQvQd.xlsx", engine="calamine")
    return df.set_index("DATE")


def test_hand_checked_rgdp_quarter(raw_routput):
    # Outcome for t = 2010Q3: x_2011Q2 / x_2010Q2 from the vintage dated 2011Q3.
    num = float(raw_routput.loc["2011:Q2", "ROUTPUT11Q3"])
    den = float(raw_routput.loc["2010:Q2", "ROUTPUT11Q3"])
    assert rt.outcome("RGDP", P("2010Q3")) == pytest.approx(100 * (num / den - 1), abs=1e-9)


def test_history_from_single_vintage(raw_routput):
    s = P("2010Q3")
    h = rt.history("RGDP", s, n=40)
    col = raw_routput["ROUTPUT10Q3"].dropna().astype(float)
    assert h.index[-1] == s - 1 and len(h) == 40
    last = 100 * (col["2010:Q2"] / col["2009:Q2"] - 1)
    first = 100 * (col["2000:Q3"] / col["1999:Q3"] - 1)
    assert h.iloc[-1] == pytest.approx(last) and h.iloc[0] == pytest.approx(first)


def test_outcome_growth_within_one_vintage():
    m = rt.vintage_matrix("PGDP")
    t = P("2015Q2")
    v = t + 4
    expected = 100 * (m.at[t + 3, v] / m.at[t - 1, v] - 1)
    assert rt.outcome("PGDP", t) == pytest.approx(expected)


def test_no_ratio_spans_rebasing():
    """The 2013 comprehensive revision rebased real GDP (vintage 2013Q4 vs 2013Q3). A growth
    rate built across vintages would be off by several percent; ours never is."""
    m = rt.vintage_matrix("RGDP")
    t = P("2013Q1")  # outcome vintage 2014Q1 (new base); x_{t-1} in vintage t (old base)
    within = rt.outcome("RGDP", t)
    across = 100 * (m.at[t + 3, t + 4] / m.at[t - 1, t] - 1)
    assert abs(within) < 6
    assert abs(across - within) > 3


def test_history_raises_if_vintage_is_stale(monkeypatch):
    s = P("2010Q3")
    real = rt.series_in_vintage

    def stale(var, vintage, **kw):
        return real(var, vintage, **kw).iloc[:-1]

    monkeypatch.setattr(rt, "series_in_vintage", stale)
    with pytest.raises(ValueError):
        rt.history("RGDP", s)


def test_rates_known_only_before_vintage():
    s = P("2008Q4")
    assert np.isfinite(rt.value("TB3M", s - 1, s))
    assert np.isnan(rt.value("TB3M", s, s))
    assert rt.history("TB3M", s).index[-1] == s - 1


def test_monthly_vintage_month_selection():
    mid = rt.vintage_matrix("INDPROD", month_in_quarter=2)
    first = rt.vintage_matrix("INDPROD", month_in_quarter=1)
    q = P("2005Q3")
    # Both have quarterly vintages, built from August (mid) and July (first) files.
    assert q in mid.columns and q in first.columns
    assert mid[q].last_valid_index() >= first[q].last_valid_index()


def test_missing_vintage_falls_forward():
    # The 1996Q1 NIPA vintage is missing (government shutdown); use the next vintage.
    m = rt.vintage_matrix("RGDP")
    v = rt.first_available_vintage(m, P("1996Q1"))
    assert v >= P("1996Q1") and m[v].notna().any()
