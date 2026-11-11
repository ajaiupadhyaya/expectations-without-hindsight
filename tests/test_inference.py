import warnings

import numpy as np
import pytest
import statsmodels.api as sm

from ewh.estimators.eor import _demean, ratio_estimate
from ewh.estimators.types import CovSpec, Draws
from ewh.inference.cov import fieller, fixed_b_critical
from ewh.inference.multiple import holm, holm_adjusted, mde


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(1)
    n = 600
    g = np.repeat(np.arange(60), 10)
    t = np.tile(np.arange(30), 20)
    g2 = rng.integers(0, 25, n)
    x = rng.normal(size=n) + 0.3 * np.repeat(rng.normal(size=60), 10)
    y = 0.5 * x + rng.normal(size=n) + np.repeat(rng.normal(size=60), 10)
    d = Draws(
        np.zeros((n, 1)),
        np.zeros((n, 1)),
        np.zeros(n),
        np.zeros(n),
        np.column_stack([g, t, g2]),
        ("g", "t", "g2"),
    )
    return x, y, g, t, g2, d


def _ours(panel, spec):
    x, y, *_, d = panel
    xt, yt = _demean(x, d.cell), _demean(y, d.cell)
    return ratio_estimate(xt * yt, xt**2, d, spec, weak=False)


def _sm(panel, **kw):
    x, y, *_ = panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.OLS(y, sm.add_constant(x)).fit(**kw)


def test_cr1_matches_statsmodels(panel):
    *_, g, _t, _g2, _d = panel
    ours = _ours(panel, CovSpec("cr1", ("g",)))
    ref = _sm(panel, cov_type="cluster", cov_kwds={"groups": g})
    n = len(g)
    # statsmodels adds (N-1)/(N-K); ours keeps only G/(G-1) (influence-value convention).
    assert ref.bse[1] == pytest.approx(ours.se * np.sqrt((n - 1) / (n - 2)), rel=1e-9)
    assert ref.params[1] == pytest.approx(ours.b)


def test_twoway_matches_statsmodels(panel):
    _x, _y, g, _t, g2, _d = panel
    ours = _ours(panel, CovSpec("twoway", ("g", "g2")))
    ref = _sm(panel, cov_type="cluster", cov_kwds={"groups": np.column_stack([g, g2])})
    n = len(g)
    assert ref.bse[1] == pytest.approx(ours.se * np.sqrt((n - 1) / (n - 2)), rel=1e-9)


def test_driscoll_kraay_matches_statsmodels(panel):
    _x, _y, _g, t, _g2, _d = panel
    ours = _ours(panel, CovSpec("dk", time="t", lags=3))
    ref = _sm(
        panel, cov_type="nw-groupsum", cov_kwds={"time": t, "maxlags": 3, "use_correction": False}
    )
    assert ref.bse[1] == pytest.approx(ours.se, rel=1e-9)


def test_newey_west_matches_statsmodels_on_a_time_series():
    rng = np.random.default_rng(3)
    T = 300
    e = np.convolve(rng.normal(size=T + 3), np.ones(4) / 2, "valid")
    x = rng.normal(size=T)
    y = 0.2 * x + e
    d = Draws(
        np.zeros((T, 1)), np.zeros((T, 1)), np.zeros(T), np.zeros(T), np.arange(T)[:, None], ("t",)
    )
    xt, yt = _demean(x, d.cell), _demean(y, d.cell)
    ours = ratio_estimate(xt * yt, xt**2, d, CovSpec("nw", time="t", lags=4), weak=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.OLS(y, sm.add_constant(x)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False}
        )
    assert ref.bse[1] == pytest.approx(ours.se, rel=1e-9)


def test_fieller_bounded_matches_delta_when_strong():
    (lo, hi), kind = fieller(abar=2.0, dbar=4.0, vaa=0.01, vad=0.0, vdd=0.001)
    assert kind == "bounded"
    se_delta = np.sqrt(0.01 / 16 + 0.25 * 0.001 / 16)
    assert lo == pytest.approx(0.5 - 1.96 * se_delta, abs=0.003)
    assert hi == pytest.approx(0.5 + 1.96 * se_delta, abs=0.003)


def test_fieller_unbounded_when_denominator_not_significant():
    _, kind = fieller(abar=0.5, dbar=0.1, vaa=0.01, vad=0.0, vdd=0.01)
    assert kind in ("complement", "unbounded")


def test_holm_worked_example():
    p = {"H1": 0.01, "H2": 0.04, "H3": 0.03}
    # sorted 0.01 <= 0.05/3, 0.03 <= 0.05/2? no (0.025) -> stop
    assert holm(p) == {"H1": True, "H2": False, "H3": False}
    assert holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    adj = holm_adjusted(p)
    assert adj == pytest.approx({"H1": 0.03, "H3": 0.06, "H2": 0.06})


def test_mde():
    assert mde(1.0) == pytest.approx(2.8016, abs=1e-3)


def test_fixed_b_critical_values():
    assert fixed_b_critical(0.0) == pytest.approx(1.96, abs=0.001)
    c1 = fixed_b_critical(0.05, n_sim=4000, T=200)
    c2 = fixed_b_critical(0.2, n_sim=4000, T=200)
    assert 1.96 < c1 < c2
    # Kiefer-Vogelsang (2005) Bartlett polynomial gives about 2.26 at b = 0.1.
    assert 2.12 < fixed_b_critical(0.1, n_sim=4000, T=200) < 2.42
