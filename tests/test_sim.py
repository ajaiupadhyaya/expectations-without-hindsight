import numpy as np
import pytest

from ewh.sim import ar1, benchmarks


def test_seed_namespaces_are_disjoint():
    s = ar1.seeds()
    assert len(set(s.values())) == len(s)
    assert {"pilot", "dryrun", "power", "gate", "confirm", "subsets"} <= set(s)


@pytest.mark.parametrize("rho", [0.0, 0.4, 0.8])
def test_sample_autocorrelation_matches_rho(rho):
    x = ar1.ar1_paths(rho, 4000, 60, "tests")
    a, b = x[:, :-1].ravel(), x[:, 1:].ravel()
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(rho, abs=0.01)
    assert x[:, -1].std() == pytest.approx(20 / np.sqrt(1 - rho**2), rel=0.05)


def test_random_walk_starts_at_zero_and_has_unit_root():
    x = ar1.ar1_paths(1.0, 3000, 41, "tests")
    assert (x[:, 0] == 0).all()
    assert x[:, -1].std() == pytest.approx(20 * np.sqrt(40), rel=0.05)


def test_seeds_reproduce_and_rho_streams_share_no_shocks():
    a = ar1.ar1_paths(0.4, 10, 20, "tests")
    b = ar1.ar1_paths(0.4, 10, 20, "tests")
    c = ar1.ar1_paths(0.6, 10, 20, "tests")
    d = ar1.ar1_paths(0.4, 10, 20, "tests", "other")
    assert np.array_equal(a, b)
    eps_a = a[:, 1:] - 0.4 * a[:, :-1]
    eps_c = c[:, 1:] - 0.6 * c[:, :-1]
    assert abs(np.corrcoef(eps_a.ravel(), eps_c.ravel())[0, 1]) < 0.15
    assert not np.array_equal(a, d)


def test_integer_paths():
    x = ar1.ar1_paths(0.6, 5, 41, "tests", integer=True)
    assert np.array_equal(x, np.rint(x))


def test_rational_forecast():
    hist = np.array([[1.0, 2.0, 10.0]])
    assert benchmarks.rational(hist, 0.5, 1)[0] == 5.0
    assert benchmarks.rational(hist, 0.5, 2)[0] == 2.5


def test_ls_learner_matches_hand_regression():
    rng = np.random.default_rng(0)
    hist = rng.normal(size=(3, 40)).cumsum(1)
    for i in range(3):
        b, a = np.polyfit(hist[i, :-1], hist[i, 1:], 1)
        f1 = a + b * hist[i, -1]
        assert benchmarks.ls_learner(hist, 1)[i] == pytest.approx(f1)
        assert benchmarks.ls_learner(hist, 2)[i] == pytest.approx(a + b * f1)
    _, b0 = benchmarks.ls_coefficients(hist, intercept=False)
    x, y = hist[0, :-1], hist[0, 1:]
    assert b0[0] == pytest.approx((x @ y) / (x @ x))


def test_simple_benchmarks():
    hist = np.array([[1.0, 3.0, 8.0]])
    assert benchmarks.random_walk(hist, 2)[0] == 8.0
    assert benchmarks.uncond_mean(hist, 1)[0] == 4.0


def test_string_tags_are_hashed_in_full():
    """Regression (Decision 14): tags sharing their first 8 bytes must give different streams."""
    a = ar1.ar1_paths(0.6, 3, 10, "tests", "chrono-2000")
    b = ar1.ar1_paths(0.6, 3, 10, "tests", "chrono-2001")
    c = ar1.ar1_paths(0.6, 3, 10, "tests", "gate-hist-a")
    d = ar1.ar1_paths(0.6, 3, 10, "tests", "gate-hist-b")
    assert not np.array_equal(a, b) and not np.array_equal(c, d)
    assert ar1._tag("chrono-2000") != ar1._tag("chrono-2001")
