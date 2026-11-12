"""Size and power simulations for Arm A (plan Task 3.7, D8, D13).

DGP (estimator space, no prompts): M models x 5 rho cells (0.2..1) x 24 templates (12 paraphrases
x 2 formats) x n paths per (model, template, rho), nested. The noise-free coefficient of a unit is
b = theta + eta_paraphrase, eta ~ N(0, tau^2): template is a random factor (handoff 3.3), so the
estimand is the average over the population of templates. Each unit is a planted forecaster
(ewh.sim.planted) with lam = 1/(1 + b), K draws with noise s.d. kappa x 20 at each date (noise
scaled to the series' innovations, which the pilot measures; a share of the cross-path forecast
variance would explode at rho = 1, where levels drift apart).

Template-inference candidates (plan D8):
  (a) cluster by path (conditional on the 12 templates used);
  (b) per-paraphrase pooled estimates, t-test across the N_PAR paraphrases (N_PAR - 1 df);
  (c) wild score bootstrap clustered by paraphrase (Webb six-point weights).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ewh.estimators import eor
from ewh.estimators.types import CovSpec, Draws
from ewh.sim import ar1, planted

RHOS = (0.2, 0.4, 0.6, 0.8, 1.0)
N_PAR, N_FMT = 12, 2  # paraphrases x formats (templates v2: 24 templates)
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


@dataclass(frozen=True)
class Design:
    models: int = 1
    paths: int = 100  # per model x template x rho
    k: int = 16
    theta: float = 0.0
    tau: float = 0.0
    kappa: float = 1.0  # per-draw noise s.d. in units of the innovation s.d. (20)


def _systematic_rw(n, theta, rng):
    """rho = 1 needs a non-stationary version of ewh.sim.planted.systematic (random walk)."""
    lam = 1.0 / (1.0 + theta)
    x_tm1 = rng.normal(0, 20.0 * math.sqrt(40), n)  # level after ~40 steps; irrelevant to b
    eps_t, eps_tp1 = rng.normal(0, 20.0, n), rng.normal(0, 20.0, n)
    x_t = x_tm1 + eps_t
    f_tm1 = x_tm1
    f_t = f_tm1 + lam * (x_t - f_tm1)
    return f_t, f_tm1, x_t + eps_tp1


def simulate(design: Design, rng: np.random.Generator) -> tuple[Draws, np.ndarray]:
    """Draws for one replication and the paraphrase id of each unit."""
    eta = rng.normal(0, design.tau, N_PAR) if design.tau > 0 else np.zeros(N_PAR)
    f_t, f_tm1, x, cell, par = [], [], [], [], []
    for _ in range(design.models):
        for p in range(N_PAR):
            for _f in range(N_FMT):
                b = design.theta + eta[p]
                for rho in RHOS:
                    n = design.paths
                    if rho < 1:
                        st, sm, xx = planted.systematic(n, rho, b, rng)
                    else:
                        st, sm, xx = _systematic_rw(n, b, rng)
                    sd = design.kappa * 20.0
                    f_t.append(st[:, None] + rng.normal(0, sd, (n, design.k)))
                    f_tm1.append(sm[:, None] + rng.normal(0, sd, (n, design.k)))
                    x.append(xx)
                    cell.append(np.full(n, rho))
                    par.append(np.full(n, p))
    par = np.concatenate(par)
    u = len(par)
    d = Draws(
        np.vstack(f_t),
        np.vstack(f_tm1),
        np.concatenate(x),
        np.concatenate(cell),
        np.column_stack([np.arange(u), par]),
        ("path", "paraphrase"),
    )
    return d, par


def pooled(d: Draws, fn=eor.split_iv):
    cov = CovSpec("cr1", ("path",))
    return eor.pooled_equal_weight(eor.by_cell(fn, d, cov), d, cov, RHOS)


def test_a(d: Draws) -> tuple[float, float]:
    e = pooled(d)
    return e.b, e.p


def test_b(d: Draws, par: np.ndarray) -> tuple[float, float]:
    bs = np.array([pooled(d.subset(par == p)).b for p in range(N_PAR)])
    t = bs.mean() / (bs.std(ddof=1) / math.sqrt(N_PAR))
    return float(bs.mean()), float(2 * stats.t.sf(abs(t), N_PAR - 1))


def test_c(d: Draws, par: np.ndarray, rng: np.random.Generator, n_boot: int = 999) -> tuple:
    """Wild score bootstrap with the null imposed (restricted scores). Under H0: b = 0 every cell
    coefficient b_c = sum(a)/sum(den) is 0, so the pooled equal-weight estimator's score is
    s_i = a_i / (C * sum_c den) for unit i in cell c. Paraphrase sums S_g are multiplied by Webb
    weights; the statistic is sum(S) / sqrt(sum(S^2) G/(G-1)) in both the sample and the draws."""
    m = eor.unit_moments(d)
    G = N_PAR
    scores = np.zeros(m.d.n)
    for c in RHOS:
        sel = m.d.cell == c
        mc = eor.UnitMoments(
            m.d.subset(sel),
            m.ebar[sel],
            m.rbar[sel],
            m.s2_t[sel],
            m.s2_tm1[sel],
            m.k_t[sel],
            m.k_tm1[sel],
        )
        a, den = eor.family_terms(mc, math.inf)
        scores[sel] = a / (len(RHOS) * den.sum())
    g = par[m.d.uid]
    S = np.bincount(g, weights=scores, minlength=G)
    t0 = S.sum() / math.sqrt((S**2).sum() * G / (G - 1))
    w = WEBB[rng.integers(0, 6, (n_boot, G))]
    Sb = w * S
    tb = Sb.sum(1) / np.sqrt((Sb**2).sum(1) * G / (G - 1))
    return float(pooled(d).b), float((np.abs(tb) >= abs(t0)).mean())


def size_table(design: Design, reps: int, seed_tag: str) -> dict:
    """Rejection rates at 5% under the null (theta = 0) for (a), (b), (c)."""
    rng = ar1.stream("power", "size", seed_tag)
    rej = np.zeros(3)
    for _ in range(reps):
        d, par = simulate(design, rng)
        rej += [test_a(d)[1] < 0.05, test_b(d, par)[1] < 0.05, test_c(d, par, rng)[1] < 0.05]
    return dict(zip(("a_path", "b_paraphrase_t", "c_wild_paraphrase"), rej / reps, strict=True))


POOLINGS = {
    "equal_02_1": {r: 1.0 for r in RHOS},
    "equal_04_1": {r: 1.0 for r in RHOS if r >= 0.4},
    "rho2_02_1": {r: r**2 for r in RHOS},
}


def robust_sd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def pooling_grid(design: Design, seed_tag: str, reps: int = 30) -> dict:
    """Per pooling scheme: median and robust s.d. of the pooled split IV and of the H1 gap, and
    the share of replications whose weakest pooled cell has first-stage t^2 below 10."""
    rng = ar1.stream("power", "pool", seed_tag)
    cov = CovSpec("cr1", ("path",))
    out = {k: dict(iv=[], gap=[], weak=0) for k in POOLINGS}
    for _ in range(reps):
        d, _ = simulate(design, rng)
        iv = eor.by_cell(eor.split_iv, d, cov)
        nv = eor.by_cell(eor.naive_stacked, d, cov)
        for name, w in POOLINGS.items():
            a = eor.pooled_weighted(iv, d, cov, w)
            b = eor.pooled_weighted(nv, d, cov, w)
            out[name]["iv"].append(a.b)
            out[name]["gap"].append(b.b - a.b)
            out[name]["weak"] += min(iv[c].first_stage_t2 for c in w) < 10
    return {
        k: dict(
            iv_median=float(np.median(v["iv"])),
            iv_rsd=robust_sd(v["iv"]),
            gap_rsd=robust_sd(v["gap"]),
            weak_share=v["weak"] / reps,
        )
        for k, v in out.items()
    }


def se_per_unit(design: Design, seed_tag: str, reps: int = 20) -> dict:
    """Monte Carlo standard deviation of the pooled IV, naive and H1 gap at this design."""
    rng = ar1.stream("power", "se", seed_tag)
    iv, nv, gap = [], [], []
    for _ in range(reps):
        d, _ = simulate(design, rng)
        a, b = pooled(d), pooled(d, eor.naive_stacked)
        iv.append(a.b)
        nv.append(b.b)
        gap.append(b.b - a.b)
    return dict(
        iv_sd=float(np.std(iv, ddof=1)),
        naive_sd=float(np.std(nv, ddof=1)),
        gap_sd=float(np.std(gap, ddof=1)),
        iv_mean=float(np.mean(iv)),
        naive_mean=float(np.mean(nv)),
    )


def mde(sd: float, df: int | None = None) -> float:
    if df is None:
        return (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd
    return (stats.t.ppf(0.975, df) + stats.t.ppf(0.8, df)) * sd
