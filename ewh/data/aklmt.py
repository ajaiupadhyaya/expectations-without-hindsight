"""AKLMT (2023 QJE) Experiment 1: the human benchmark for Arm A.

Reproduces the authors' Figure II (code/FigureII.do in their replication package):
- implied persistence: OLS of the one-step forecast on last period's realization, by rho;
- error-on-revision: OLS of (actual - one-step forecast) on (one-step forecast minus the
  previous round's two-step forecast of the same target), by rho, clustered by participant.
`python -m ewh.data.aklmt` writes data/processed/aklmt_figure2.csv.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ewh.provenance.archive import RAW_ROOT, REPO_ROOT

ZIP_PATH = RAW_ROOT / "aklmt" / "Replication.zip"
MEMBER = "Afrouzi replication/data/Experiment1/detail.csv"
RHOS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
OUT = REPO_ROOT / "data" / "processed" / "aklmt_figure2.csv"


def load_experiment1(zip_path: Path = ZIP_PATH) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(io.BytesIO(z.read(MEMBER)))
    df["rho"] = df["rho"].round(1)
    df = df.sort_values(["assignmentid", "round"]).reset_index(drop=True)
    g = df.groupby("assignmentid")
    # Stata `xtset id round`: a lag exists only when the previous round is exactly round - 1.
    consecutive = g["round"].diff() == 1
    df["actual_lag"] = g["actual"].shift(1).where(consecutive)
    df["predb_lag"] = g["predictionb_value"].shift(1).where(consecutive)
    df["error"] = df["actual"] - df["predictiona_value"]
    df["revision"] = df["predictiona_value"] - df["predb_lag"]
    return df


def ols_cluster(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict:
    """OLS of y on [1, x] with participant-clustered errors.

    `se` matches Stata ivreg2's cluster option (no small-sample factor); `se_cr1` adds the
    usual G/(G-1)·(N-1)/(N-K) factor.
    """
    X = np.column_stack([np.ones_like(x), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    codes, idx = np.unique(groups, return_inverse=True)
    scores = np.zeros((len(codes), 2))
    np.add.at(scores, idx, X * u[:, None])
    vcov = xtx_inv @ (scores.T @ scores) @ xtx_inv
    n, k, g = len(y), 2, len(codes)
    factor = g / (g - 1) * (n - 1) / (n - k)
    return {
        "b": float(beta[1]),
        "se": float(np.sqrt(vcov[1, 1])),
        "se_cr1": float(np.sqrt(vcov[1, 1] * factor)),
        "n": n,
        "clusters": g,
    }


def figure2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rho in RHOS:
        d = df[np.isclose(df["rho"], rho)]
        p = d.dropna(subset=["predictiona_value", "actual_lag"])
        ip = ols_cluster(
            p["predictiona_value"].to_numpy(float),
            p["actual_lag"].to_numpy(float),
            p["assignmentid"].to_numpy(),
        )
        c = d.dropna(subset=["error", "revision"])
        cg = ols_cluster(
            c["error"].to_numpy(float), c["revision"].to_numpy(float), c["assignmentid"].to_numpy()
        )
        rows.append(
            {
                "rho": rho,
                "implied_rho": ip["b"],
                "implied_rho_se": ip["se"],
                "b_eor": cg["b"],
                "b_eor_se": cg["se"],
                "b_eor_se_cr1": cg["se_cr1"],
                "n_eor": cg["n"],
                "participants": cg["clusters"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    table = figure2(load_experiment1())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.6f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
