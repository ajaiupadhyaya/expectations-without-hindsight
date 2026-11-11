import itertools
import math

import numpy as np
import pytest

from ewh.estimators import eor
from ewh.estimators.persistence import implied_persistence, noise_share
from ewh.estimators.types import CovSpec, Draws
from ewh.sim import ar1, benchmarks, planted

UNIT = CovSpec("cr1", ("path",))
RHO = 0.6


def rng(*tags):
    return ar1.stream("tests", *tags)


def test_naive_matches_general_formula():
    sd_t, sd_tm1, theta = 15.0, 25.0, -0.3
    d = planted.planted_draws(200_000, RHO, theta, sd_t, sd_tm1, 8, rng("formula"))
    target = planted.naive_limit(RHO, theta, sd_t, sd_tm1)
    for est in (eor.naive_stacked(d, UNIT), eor.naive_single(d, UNIT)):
        assert est.b == pytest.approx(target, abs=4 * est.se)
    assert eor.split_iv(d, UNIT).b == pytest.approx(theta, abs=0.02)


def test_attraction_point():
    """Noise pulls the one-draw coefficient toward -s_t^2/(s_t^2 + s_tm1^2) (= -1/2 with equal
    noise) from either side: rational and underreacting forecasters are pulled down, a strong
    overreactor is pulled up (plan change 2)."""
    s = 40.0
    for theta in (-1.5, -0.5, 0.0, 0.3):
        d = planted.planted_draws(100_000, RHO, theta, s, s, 8, rng("attr", str(theta)))
        e = eor.naive_stacked(d, UNIT)
        assert e.b == pytest.approx(planted.naive_limit(RHO, theta, s, s), abs=4 * e.se)
        if theta == -0.5:
            assert e.b == pytest.approx(-0.5, abs=4 * e.se)
        else:
            assert np.sign(e.b - theta) == np.sign(-0.5 - theta)
            assert abs(e.b - theta) > 6 * e.se


def _random_draws(n, k, cells, seed):
    g = np.random.default_rng(seed)
    base = g.normal(size=n) * 10
    return Draws(
        f_t=base[:, None] + g.normal(size=(n, k)) * 3,
        f_tm1=base[:, None] * 0.5 + g.normal(size=(n, k)) * 4,
        x=base * 0.8 + g.normal(size=n) * 5,
        cell=np.array(cells),
        clusters=np.arange(n)[:, None],
        cluster_names=("path",),
    )


def test_split_iv_equals_enumeration():
    d = _random_draws(8, 4, [0, 0, 0, 0, 1, 1, 1, 1], seed=7)
    closed = eor.b_of_k(eor.unit_moments(d, min_valid=2), math.inf, UNIT).b
    assert eor.split_iv_enumerated(d) == pytest.approx(closed, abs=1e-10)


def test_cross_unit_averaging_by_brute_force():
    """Two units in one cell: average the demeaned IV sums over all 36^2 joint splits."""
    d = _random_draws(2, 4, [0, 0], seed=11)
    halves = list(itertools.combinations(range(4), 2))
    num = den = 0.0
    count = 0
    for s in itertools.product(halves, repeat=4):  # (t, t-1) splits for unit 0 and unit 1
        ys, rs, zs = [], [], []
        for i in range(2):
            ta, ma = list(s[2 * i]), list(s[2 * i + 1])
            tb = [j for j in range(4) if j not in ta]
            mb = [j for j in range(4) if j not in ma]
            fa, fb = d.f_t[i, ta].mean(), d.f_t[i, tb].mean()
            ga, gb = d.f_tm1[i, ma].mean(), d.f_tm1[i, mb].mean()
            ys.append(d.x[i] - fa)
            rs.append(fa - ga)
            zs.append(fb - gb)
        y, r, z = map(np.array, (ys, rs, zs))
        num += ((z - z.mean()) * (y - y.mean())).sum()
        den += ((z - z.mean()) * (r - r.mean())).sum()
        count += 1
    brute = (num / count) / (den / count)
    closed = eor.b_of_k(eor.unit_moments(d, min_valid=2), math.inf, UNIT).b
    assert brute == pytest.approx(closed, abs=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("theta", [-0.7, -0.3, 0.0, 0.3])
@pytest.mark.parametrize("share", [0.0, 0.3, 0.6])
def test_split_iv_coverage(theta, share):
    """Handoff's required test: split-sample IV recovers the planted coefficient, with 95%
    intervals covering it in 93-97% of 2,000 replications."""
    sd = planted.noise_sd_for_share(share, RHO, theta) if share > 0 else 0.0
    g = rng("coverage", str(theta), str(share))
    reps, hits, bs = 2000, 0, []
    for _ in range(reps):
        d = planted.planted_draws(1000, RHO, theta, sd, sd, 8, g)
        e = eor.split_iv(d, UNIT)
        hits += e.ci[0] <= theta <= e.ci[1]
        bs.append(e.b)
    assert 0.93 <= hits / reps <= 0.97, hits / reps
    assert np.mean(bs) == pytest.approx(theta, abs=0.02)


def test_mean_of_k_tracks_b_of_k():
    sd = 25.0
    d = planted.planted_draws(100_000, RHO, -0.3, sd, sd, 16, rng("ladder"))
    m = eor.unit_moments(d)
    for k in (1, 2, 4, 8):
        fam = eor.b_of_k(m, k, UNIT)
        sub = eor.mean_of_k(d, k, UNIT, n_subsets=10, seed=k)
        assert sub.b == pytest.approx(fam.b, abs=0.02), k
    assert eor.b_of_k(m, 16, UNIT).b == pytest.approx(eor.mean_of_k(d, 16, UNIT, 1).b, abs=1e-9)


def test_zero_draw_variance_is_finite():
    d = planted.planted_draws(500, RHO, -0.3, 0.0, 0.0, 8, rng("zero"))
    e = eor.split_iv(d, UNIT)
    assert np.isfinite(e.b) and np.isfinite(e.se) and e.se > 0
    # With identical draws the correction terms are zero: IV equals the one-draw regression.
    assert e.b == pytest.approx(eor.naive_single(d, UNIT).b, abs=1e-12)
    assert noise_share(d) == 0.0


def test_weak_first_stage_flagged():
    d = planted.planted_draws(200, 0.2, 0.0, 60.0, 60.0, 8, rng("weak"))
    e = eor.split_iv(d, UNIT)
    assert e.first_stage_t2 < 10
    lo, hi = e.fieller
    assert e.fieller_kind != "bounded" or hi - lo > 1.0


def _ls_draws(rho, n, k=8):
    x = ar1.ar1_paths(rho, n, 42, "tests", "ls")
    f_tm1 = benchmarks.ls_learner(x[:, :40], 2)
    f_t = benchmarks.ls_learner(x[:, :41], 1)
    d = Draws(
        np.repeat(f_t[:, None], k, 1),
        np.repeat(f_tm1[:, None], k, 1),
        x[:, 41],
        np.full(n, rho),
        np.arange(n)[:, None],
        ("path",),
    )
    return d, x[:, 40]


def test_ls_learner_benchmark_values():
    """The least-squares learner fitted to 40 shown values (Phase 1 finding, decisions.md):
    its error-on-revision coefficient is clearly nonzero and changes sign with rho, so H2 needs
    the LS contrast; its implied persistence is close to unbiased at every rho."""
    rhos = (0.2, 0.4, 0.6, 0.8, 1.0)
    b, gaps = {}, []
    for rho in rhos:
        d, x_last = _ls_draws(rho, 40_000)
        b[rho] = eor.split_iv(d, UNIT).b
        gaps.append(implied_persistence(d, x_last, UNIT).b - rho)
    assert b[0.2] < -0.2
    assert b[1.0] > 0.05
    assert max(abs(g) for g in gaps) < 0.02
    assert abs(np.polyfit(rhos, gaps, 1)[0]) < 0.03


def test_common_sample():
    d = planted.planted_draws(400, RHO, -0.3, 10.0, 10.0, 10, rng("common"))
    f_t = d.f_t.copy()
    f_t[:50, :4] = np.nan  # 6 valid draws: below min_valid = 8, dropped
    f_t[50:80, 0] = np.nan  # 9 valid: kept, and naive uses the first valid draw
    d = Draws(f_t, d.f_tm1, d.x, d.cell, d.clusters, d.cluster_names)
    uids = [
        e.uid
        for e in (
            eor.split_iv(d, UNIT),
            eor.naive_single(d, UNIT),
            eor.naive_stacked(d, UNIT),
            eor.mean_of_k(d, 4, UNIT, n_subsets=3),
        )
    ]
    for u in uids[1:]:
        assert np.array_equal(u, uids[0])
    assert len(uids[0]) == 350


def test_difference_and_equal_weight_pooling():
    parts = [
        planted.planted_draws(5000, RHO, th, 20.0, 20.0, 8, rng("pool", str(th)), cell=c)
        for c, th in enumerate((-0.3, 0.0))
    ]
    d = Draws(
        np.vstack([p.f_t for p in parts]),
        np.vstack([p.f_tm1 for p in parts]),
        np.concatenate([p.x for p in parts]),
        np.concatenate([p.cell for p in parts]),
        np.arange(10_000)[:, None],
        ("path",),
    )
    by = eor.by_cell(eor.split_iv, d, UNIT)
    pooled = eor.pooled_equal_weight(by, d, UNIT)
    assert pooled.b == pytest.approx(np.mean([e.b for e in by.values()]))
    assert pooled.se == pytest.approx(np.sqrt(sum(e.se**2 for e in by.values())) / 2, rel=0.02)
    naive_by = eor.by_cell(eor.naive_stacked, d, UNIT)
    gap = eor.difference(eor.pooled_equal_weight(naive_by, d, UNIT), pooled, d, UNIT, label="H1")
    assert gap.b < 0 and gap.ci[1] < 0  # noise pulls these toward -1/2
