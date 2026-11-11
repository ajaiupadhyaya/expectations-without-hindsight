"""Shared data types for the estimator family (plan "Core interfaces")."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Draws:
    """One row per unit (a revision pair: one path, template and model, or one SPF quarter).

    f_t, f_tm1: (U, K) forecast draws of the same target made at t and t-1; NaN = invalid.
    x: (U,) realized target. cell: (U,) cell id (rho, or variable); intercepts are per cell.
    clusters: (U, C) integer ids, columns named by `cluster_names`.
    uid: (U,) unit ids in the parent table, kept through subsetting so estimates align.
    """

    f_t: np.ndarray
    f_tm1: np.ndarray
    x: np.ndarray
    cell: np.ndarray
    clusters: np.ndarray
    cluster_names: tuple[str, ...]
    uid: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.uid is None:
            object.__setattr__(self, "uid", np.arange(len(self.x)))
        if self.clusters.ndim == 1:
            object.__setattr__(self, "clusters", self.clusters[:, None])
        u = len(self.x)
        assert self.f_t.shape[0] == u and self.f_tm1.shape[0] == u
        assert len(self.cell) == u and self.clusters.shape == (u, len(self.cluster_names))

    @property
    def n(self) -> int:
        return len(self.x)

    def column(self, name: str) -> np.ndarray:
        if name == "unit":
            return self.uid
        return self.clusters[:, self.cluster_names.index(name)]

    def subset(self, mask: np.ndarray) -> Draws:
        return Draws(
            self.f_t[mask],
            self.f_tm1[mask],
            self.x[mask],
            self.cell[mask],
            self.clusters[mask],
            self.cluster_names,
            self.uid[mask],
        )


@dataclass(frozen=True)
class CovSpec:
    """How to turn unit-level influence values into a variance.

    kind: "cr1" (one-way clusters; "unit" means heteroskedasticity-robust), "twoway"
    (Cameron-Gelbach-Miller on two cluster columns), "dk" (Driscoll-Kraay: sum within the
    `time` column, then Bartlett HAC with `lags`), "nw" (alias of dk for one unit per period).
    """

    kind: str = "cr1"
    clusters: tuple[str, ...] = ("unit",)
    time: str | None = None
    lags: int | None = None


@dataclass(frozen=True)
class Estimate:
    b: float
    se: float
    ci: tuple[float, float]
    p: float
    n_units: int
    n_clusters: tuple[int, ...]
    psi: np.ndarray  # (n_units,) influence values: b_hat - b ~ mean(psi)
    uid: np.ndarray  # (n_units,) unit ids aligned with psi
    label: str = ""
    first_stage_t2: float | None = None
    fieller: tuple[float, float] | None = None
    fieller_kind: str | None = None  # "bounded", "complement" (two rays) or "unbounded"
