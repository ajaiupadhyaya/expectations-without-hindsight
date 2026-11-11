"""Noise-ladder Monte Carlo for planted forecasters (plan Task 1.4, `make noise-mc`).

For planted coefficients theta and a fixed noise share, estimates b(k) for k = 1..16 and the
split-sample IV (k = inf), and draws them against the theoretical curve
(C - s_t^2/k) / (V + S/k). Writes data/processed/noise_ladder_planted.csv and
paper/figures/noise_ladder_planted.pdf.
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ewh.estimators import eor  # noqa: E402
from ewh.estimators.types import CovSpec  # noqa: E402
from ewh.provenance.archive import REPO_ROOT  # noqa: E402
from ewh.sim import ar1, planted  # noqa: E402

RHO, SHARE, K, N = 0.6, 0.5, 16, 50_000
THETAS = (-0.8, -0.5, 0.0, 0.3)
KS = (1, 2, 4, 8, 16)
CSV = REPO_ROOT / "data" / "processed" / "noise_ladder_planted.csv"
FIG = REPO_ROOT / "paper" / "figures" / "noise_ladder_planted.pdf"


def theory(theta: float, sd: float, k: float) -> float:
    V = planted.var_revision(RHO, theta)
    s2 = sd**2
    inv = 0.0 if math.isinf(k) else 1 / k
    return (theta * V - s2 * inv) / (V + 2 * s2 * inv)


def run() -> pd.DataFrame:
    cov = CovSpec("cr1", ("path",))
    rows = []
    for theta in THETAS:
        sd = planted.noise_sd_for_share(SHARE, RHO, theta)
        d = planted.planted_draws(
            N, RHO, theta, sd, sd, K, ar1.stream("dryrun", "noise-mc", str(theta))
        )
        m = eor.unit_moments(d)
        for k in (*KS, math.inf):
            e = eor.b_of_k(m, k, cov)
            rows.append(
                dict(theta=theta, k=k, b=e.b, lo=e.ci[0], hi=e.ci[1], theory=theory(theta, sd, k))
            )
        rows.append(
            dict(
                theta=theta,
                k=0,
                b=eor.naive_single(d, cov).b,
                lo=np.nan,
                hi=np.nan,
                theory=theory(theta, sd, 1),
            )
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    xs = list(range(len(KS) + 1))
    labels = [str(k) for k in KS] + ["IV"]
    for theta in THETAS:
        s = df[(df.theta == theta) & (df.k != 0)].sort_values("k")
        (line,) = ax.plot(xs, s.theory, lw=1, alpha=0.6)
        ax.errorbar(
            xs,
            s.b,
            yerr=[s.b - s.lo, s.hi - s.b],
            fmt="o",
            ms=4,
            color=line.get_color(),
            label=f"planted b* = {theta:g}",
        )
    ax.axhline(-0.5, color="0.5", ls=":", lw=1)
    ax.set_xticks(xs, labels)
    ax.set_xlabel("draws averaged per prompt (k)")
    ax.set_ylabel("error-on-revision coefficient")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)


def main() -> int:
    df = run()
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.6f")
    plot(df)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
