import pandas as pd
import pytest

from ewh.data import pairs

P = pd.Period


def test_first_pair_is_2000Q2():
    assert P("2000Q2") == pairs.FIRST_T
    assert pairs.assign(P("2000Q2")) == (1999, 1999)
    with pytest.raises(AssertionError):
        assert pairs.valid_vintage(P("2000Q1") - 1) >= 1999  # 1999Q4 would need vintage 1998


def test_pair_never_mixes_vintages():
    for t in pd.period_range("2000Q2", "2025Q4", freq="Q"):
        a, b = pairs.assign(t, "primary")
        assert a == b
        assert a <= t.year - 1 and a <= (t - 1).year - 1


def test_q1_uses_lagged_vintage():
    # t = 2010Q1: the t-1 survey is 2009Q4, so both forecasts use the 2008 vintage.
    assert pairs.assign(P("2010Q1")) == (2008, 2008)
    assert pairs.assign(P("2010Q1"), "latest") == (2008, 2009)
    assert pairs.assign(P("2010Q2")) == (2009, 2009)


def test_failed_vintage_substitution():
    passing = [1999, 2003, 2010]
    assert pairs.substitute(2005, passing) == 2003
    assert pairs.substitute(2010, passing) == 2010
    assert pairs.substitute(1999, [2003]) is None  # never substitute a later (leaky) vintage


def test_six_vintage_fallback_assignment():
    assert pairs.six_vintage_fallback(1999) == 1999
    assert pairs.six_vintage_fallback(2008) == 2004
    assert pairs.six_vintage_fallback(2024) == 2024


def test_contaminated_flag():
    # 2023Q4: target 2024Q3 published in vintage 2024Q4 (15 Nov 2024) -> seen by the 2024 model.
    assert pairs.contaminated(P("2023Q4"))
    # 2024Q1: target 2024Q4 first published in vintage 2025Q1 (15 Feb 2025) -> clean.
    assert not pairs.contaminated(P("2024Q1"))
    assert pairs.vintage_date(P("2025Q1")) == pd.Timestamp("2025-02-15")
