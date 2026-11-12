"""SPF replication check (plan Task 2.4; tolerance fixed in docs/decisions.md, Decision 7).

Reference (i): BGMS Table 3 coefficients computed from BGMS's own processed panel
(spf_out_trim5.dta in their package). Ours: the same regressions on our pipeline built from raw
SPF and RTDSM files (2026-11-11 snapshot). Also the CG (2015) inflation coefficient.
Writes data/processed/spf_replication.csv and docs/spf_replication.md.
`python -m ewh.data.spf_replication`
"""

from __future__ import annotations

import io
import sys
import zipfile

import numpy as np
import pandas as pd

from ewh.data import spf
from ewh.estimators.ols import nw_auto_lags, slope
from ewh.estimators.types import CovSpec
from ewh.provenance.archive import RAW_ROOT, REPO_ROOT

BGMS_ZIP = RAW_ROOT / "bgms" / "118422-V1.zip"
CG_ZIP = RAW_ROOT / "cg" / "112862-V1.zip"
HEADLINE = ("NGDP", "RGDP", "PGDP", "CPI", "UNEMP", "TB3M", "TN10Y")
ALL = (*HEADLINE, "RCONSUM", "RNRESIN", "RRESINV", "RGF", "RGSL", "HOUSING", "INDPROD")
BGMS_NAME = {"TB3M": "tb3m", "TN10Y": "tn10y"}
# Aug 2019 revision, Table 3 (docs/references_verified.md): consensus, individual (no FE).
PUBLISHED_2019 = {
    "NGDP": (0.56, -0.22),
    "RGDP": (0.44, -0.15),
    "PGDP": (1.41, 0.18),
    "CPI": (0.29, -0.19),
    "UNEMP": (0.82, 0.34),
    "TB3M": (0.60, 0.27),
    "TN10Y": (-0.02, -0.19),
}
FIRST, LAST = pd.Period("1968Q4"), pd.Period("2016Q4")
CG_FIRST, CG_LAST = pd.Period("1969Q1"), pd.Period("2014Q4")
TOL, MIN_SIGN = 0.15, 6
CSV = REPO_ROOT / "data" / "processed" / "spf_replication.csv"
MD = REPO_ROOT / "docs" / "spf_replication.md"


def _regressions(s: pd.DataFrame) -> dict:
    """Consensus (Newey-West, auto lags) and pooled individual (two-way clustered) slopes."""
    c = spf.consensus(s).dropna()
    T = len(c)
    t_idx = np.arange(T)
    cons = slope(
        c.err.to_numpy(float),
        c.rev.to_numpy(float),
        {"t": t_idx},
        CovSpec("nw", time="t", lags=nw_auto_lags(T)),
    )
    ind = s.dropna(subset=["err", "rev"])
    q = ind.t.map(lambda p: p.ordinal).to_numpy()
    indiv = slope(
        ind.err.to_numpy(float),
        ind.rev.to_numpy(float),
        {"ID": ind.ID.to_numpy(), "t": q},
        CovSpec("twoway", ("ID", "t")),
    )
    return dict(
        cons_b=cons.b, cons_se=cons.se, T=T, ind_b=indiv.b, ind_se=indiv.se, N=indiv.n_units
    )


_PANEL: dict = {}


def bgms_panel(var: str) -> pd.DataFrame:
    """BGMS's own errors and revisions for one variable (their processed panel, year <= 2016)."""
    if "raw" not in _PANEL:
        with zipfile.ZipFile(BGMS_ZIP) as z:
            raw = pd.read_stata(io.BytesIO(z.read("programs/data/spf/spf_out_trim5.dta")))
        _PANEL["raw"] = raw[raw.year <= 2016]
    raw = _PANEL["raw"]
    name = BGMS_NAME.get(var, var)
    return pd.DataFrame(
        {
            "ID": raw.ID.astype(int),
            "t": pd.PeriodIndex(pd.to_datetime(raw.yq), freq="Q"),
            "err": raw[f"e{name}_f3ql"],
            "rev": raw[f"r{name}_f3q"],
        }
    )


def bgms_reference() -> dict[str, dict]:
    return {var: _regressions(spf.sample(bgms_panel(var), FIRST, LAST)) for var in ALL}


def ours() -> dict[str, dict]:
    return {var: _regressions(spf.sample(spf.spf_pairs(var, "bgms"), FIRST, LAST)) for var in ALL}


def cg_check() -> dict:
    """CG inflation coefficient: our PGDP consensus on CG's sample, and CG's own data."""
    s = spf.sample(spf.spf_pairs("PGDP", "bgms"), CG_FIRST, CG_LAST)
    mine = _regressions(s)
    with zipfile.ZipFile(CG_ZIP) as z:
        d = pd.read_stata(io.BytesIO(z.read("Replication_files/workfiles/us_data_updated.dta")))
    d = d[(d.forecsource == "spf") & (d.forvar == "pi")]
    w = d.pivot_table(index="timeseq", columns="fehor", values=["ind_fe", "ind_for"])
    fe = (np.prod([w["ind_fe"][h] / 400 + 1 for h in range(4)], axis=0) - 1) * 100
    f30 = (np.prod([w["ind_for"][h] / 400 + 1 for h in range(4)], axis=0) - 1) * 100
    f41 = (np.prod([w["ind_for"][h] / 400 + 1 for h in range(1, 5)], axis=0) - 1) * 100
    fr = f30 - pd.Series(f41, index=w.index).shift(1).to_numpy()
    ok = np.isfinite(fe) & np.isfinite(fr)
    T = int(ok.sum())
    theirs = slope(
        np.asarray(fe)[ok],
        np.asarray(fr)[ok],
        {"t": np.arange(T)},
        CovSpec("nw", time="t", lags=4),
    )
    ref = _regressions(spf.sample(bgms_panel("PGDP"), CG_FIRST, CG_LAST))
    return dict(
        ours_b=mine["cons_b"],
        ours_se=mine["cons_se"],
        T=mine["T"],
        ref_b=ref["cons_b"],
        ref_T=ref["T"],
        cg_data_b=theirs.b,
        cg_data_se=theirs.se,
        cg_T=T,
    )


def verdict(ref: dict, mine: dict, cg: dict) -> tuple[bool, list[str]]:
    lines = []
    within = [abs(mine[v]["cons_b"] - ref[v]["cons_b"]) <= TOL for v in HEADLINE]
    signs = [np.sign(mine[v]["cons_b"]) == np.sign(ref[v]["cons_b"]) for v in HEADLINE]
    cg_ok = 1.0 <= cg["ours_b"] <= 1.5
    lines.append(f"consensus within {TOL} of reference (i): {sum(within)}/{len(HEADLINE)}")
    lines.append(f"same sign as reference (i): {sum(signs)}/{len(HEADLINE)} (need {MIN_SIGN})")
    lines.append(f"CG inflation coefficient in [1.0, 1.5]: {cg_ok} ({cg['ours_b']:.3f})")
    return all(within) and sum(signs) >= MIN_SIGN and cg_ok, lines


def write(ref: dict, mine: dict, cg: dict) -> bool:
    rows = []
    for v in ALL:
        pub = PUBLISHED_2019.get(v, (np.nan, np.nan))
        rows.append(
            dict(
                variable=v,
                headline=v in HEADLINE,
                cons_ref=ref[v]["cons_b"],
                cons_ours=mine[v]["cons_b"],
                cons_ours_se=mine[v]["cons_se"],
                cons_pub2019=pub[0],
                ind_ref=ref[v]["ind_b"],
                ind_ours=mine[v]["ind_b"],
                ind_ours_se=mine[v]["ind_se"],
                ind_pub2019=pub[1],
                T_ref=ref[v]["T"],
                T_ours=mine[v]["T"],
                N_ref=ref[v]["N"],
                N_ours=mine[v]["N"],
            )
        )
    df = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.6f")
    ok, lines = verdict(ref, mine, cg)

    def f(x):
        return "" if not np.isfinite(x) else f"{x:.2f}"

    md = [
        "# SPF replication check (Phase 2.4)",
        "",
        "Generated by `python -m ewh.data.spf_replication`; do not edit by hand. Tolerance and",
        "rules fixed in `docs/decisions.md`, Decision 7, before this script was run.",
        "",
        f"**Verdict: {'PASS' if ok else 'FAIL'}.** " + "; ".join(lines) + ".",
        "",
        "Reference (i) = BGMS Table 3 regressions run on BGMS's own processed panel",
        "(`spf_out_trim5.dta`). Ours = the same regressions on our pipeline from raw SPF and",
        "RTDSM files. Published = Aug 2019 revision of BGMS Table 3. Sample 1968Q4–2016Q4.",
        "Consensus: mean of individual errors and revisions by quarter, Newey–West standard",
        "errors. Individual: pooled, two-way clustered by forecaster and quarter.",
        "",
        "| Variable | Consensus: ref (i) | ours (se) | published | Individual: ref (i) | ours (se) "
        "| published | T ref/ours | N ref/ours |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        star = "" if r.headline else " †"
        md.append(
            f"| {r.variable}{star} | {f(r.cons_ref)} | {f(r.cons_ours)} ({f(r.cons_ours_se)}) | "
            f"{f(r.cons_pub2019)} | {f(r.ind_ref)} | {f(r.ind_ours)} ({f(r.ind_ours_se)}) | "
            f"{f(r.ind_pub2019)} | {r.T_ref}/{r.T_ours} | {r.N_ref}/{r.N_ours} |"
        )
    md += [
        "",
        "† not part of the pass/fail rule.",
        "",
        "## Coibion–Gorodnichenko inflation coefficient",
        "",
        f"Our PGDP consensus coefficient on CG's sample ({CG_FIRST}–{CG_LAST}): "
        f"{cg['ours_b']:.2f} (se {cg['ours_se']:.2f}, T = {cg['T']}). The same regression on "
        f"BGMS's own panel and sample window: {cg['ref_b']:.2f} (T = {cg['ref_T']}). "
        "CG's own processed data "
        f"(`us_data_updated.dta`, four-quarter inflation error on revision): {cg['cg_data_b']:.2f} "
        f"(se {cg['cg_data_se']:.2f}, T = {cg['cg_T']}). CG's NBER WP 16537 reports 1.23 (0.50) "
        "for 1969–2010.",
        "",
    ]
    MD.write_text("\n".join(md))
    return ok


def main() -> int:
    ref, mine, cg = bgms_reference(), ours(), cg_check()
    ok = write(ref, mine, cg)
    print(MD.read_text())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
