"""Planted forecasters with a known error-on-revision coefficient and known draw noise.

Systematic forecasts: F*_{t-1} = rho^2 x_{t-1} (rational two-step), and
F*_t = F*_{t-1} + lam (rho x_t - F*_{t-1}) with lam = 1/(1 + theta). Then
    r* = lam rho eps_t,  e* = (1 - lam) rho eps_t + eps_{t+1},  b* = (1 - lam)/lam = theta,
for any rho > 0. theta < 0 is overreaction, theta > 0 underreaction, theta = 0 rational.
Observed draws add independent N(0, sd_t^2) and N(0, sd_tm1^2) noise.
"""

from __future__ import annotations

import numpy as np

from ewh.estimators.types import Draws
from ewh.sim.ar1 import SD


def systematic(n: int, rho: float, theta: float, rng: np.random.Generator, sd: float = SD):
    """Return (F*_t, F*_tm1, x_{t+1}) for n independent units."""
    assert 0 < rho < 1
    lam = 1.0 / (1.0 + theta)
    x_tm1 = rng.normal(0, sd / np.sqrt(1 - rho**2), n)
    eps_t, eps_tp1 = rng.normal(0, sd, n), rng.normal(0, sd, n)
    x_t = rho * x_tm1 + eps_t
    x_tp1 = rho * x_t + eps_tp1
    f_tm1 = rho**2 * x_tm1
    f_t = f_tm1 + lam * (rho * x_t - f_tm1)
    return f_t, f_tm1, x_tp1


def var_systematic_t(rho: float, theta: float, sd: float = SD) -> float:
    lam = 1.0 / (1.0 + theta)
    return rho**4 * sd**2 / (1 - rho**2) + lam**2 * rho**2 * sd**2


def var_revision(rho: float, theta: float, sd: float = SD) -> float:
    lam = 1.0 / (1.0 + theta)
    return lam**2 * rho**2 * sd**2


def noise_sd_for_share(share: float, rho: float, theta: float, sd: float = SD) -> float:
    """Per-draw noise sd giving noise_share = sigma_u^2 / (sigma_u^2 + Var(F*_t))."""
    return float(np.sqrt(share / (1 - share) * var_systematic_t(rho, theta, sd)))


def planted_draws(
    n: int,
    rho: float,
    theta: float,
    sd_t: float,
    sd_tm1: float,
    k: int,
    rng: np.random.Generator,
    cell: float = 0.0,
) -> Draws:
    f_t, f_tm1, x = systematic(n, rho, theta, rng)
    draws_t = (
        f_t[:, None] + rng.normal(0, sd_t, (n, k)) if sd_t > 0 else np.repeat(f_t[:, None], k, 1)
    )
    draws_tm1 = (
        f_tm1[:, None] + rng.normal(0, sd_tm1, (n, k))
        if sd_tm1 > 0
        else np.repeat(f_tm1[:, None], k, 1)
    )
    return Draws(
        f_t=draws_t,
        f_tm1=draws_tm1,
        x=x,
        cell=np.full(n, cell),
        clusters=np.arange(n)[:, None],
        cluster_names=("path",),
    )


def naive_limit(rho: float, theta: float, sd_t: float, sd_tm1: float) -> float:
    """plim of the one-draw coefficient: (C - s_t^2) / (V + s_t^2 + s_tm1^2), C = theta V."""
    V = var_revision(rho, theta)
    return (theta * V - sd_t**2) / (V + sd_t**2 + sd_tm1**2)
