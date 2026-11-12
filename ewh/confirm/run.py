"""Confirmatory runs, one family at a time, each at most once (plan Task 4.2, 6.1).

    python -m ewh.confirm.run gate     # competence gate: who enters the confirmatory analysis
    python -m ewh.confirm.run arm_a    # H1, H2(-LS), H3 (Holm family), robustness rows
    python -m ewh.confirm.run arm_b    # H4, H5 (Holm family), secondaries, robustness rows
    python -m ewh.confirm.run dry-generate | gate --dry-run | arm_a --dry-run | arm_b --dry-run

A real run refuses unless the tag binding holds (annotated `pap-v1.2`, bound files unchanged), the
raw archive verifies, the generation ledger is intact and every pre-registered shard (and nothing
else) is present and matches the manifest. It then writes docs/results/<family>.attempt and only
after that reads any generation. Results are written once to docs/results/<family>.{md,json}; a
failure after the marker writes its traceback to docs/results/<family>.failed.md and re-raises.

Dry runs use the `dryrun` namespace, small sizes and the synthetic DryRunBackend, read from
data/dryrun_session/ and write docs/results/dryrun_<family>.*, with no binding or marker.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ewh import session
from ewh.analysis import arm_a as A
from ewh.analysis import arm_b as B
from ewh.capability import gate as G
from ewh.confirm import binding
from ewh.data import pairs
from ewh.inference.multiple import holm, holm_adjusted
from ewh.models.ledger import Ledger
from ewh.models.runner import read_draws
from ewh.provenance.archive import RAW_ROOT, REPO_ROOT

RESULTS = REPO_ROOT / "docs" / "results"
GEN = REPO_ROOT / "data" / "generations"
DRY_GEN = REPO_ROOT / "data" / "dryrun_session"
K = session.ALLOC["k"]


# ---------------------------------------------------------------- loading
def draws_for(root: Path, run: str) -> pd.DataFrame:
    # a substitution run `<sub>-for-<failed>` holds the substitute model's draws
    return read_draws(root, run).assign(model=run.split("-for-")[0])


def runs_present(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("shard_*.parquet")))


def excluded_runs(root: Path, spec: session.Spec) -> dict[str, str]:
    """Deviation 1: models the session recorded as not generated, and their ledger event.

    A ChronoGPT vintage that failed the preflight ('preflight_failed') or a ladder model that the
    session skipped ('skipped') is left out of the expected shard set and of every family, unless
    all of its shards were later completed. A preflight failure counts as failing the gate.
    """
    path = root / "ledger.jsonl"
    if not path.exists():
        return {}
    ledger = Ledger(path)
    events = {
        e["model"]: e["event"]
        for e in ledger.entries()
        if e.get("event") in ("preflight_failed", "skipped") and "model" in e
    }
    return {
        m: ev
        for m, ev in events.items()
        if len(ledger.completed(m)) != session.n_shards(session.model_table(m, spec))
    }


# ---------------------------------------------------------------- gate
def gate_family(root: Path, spec: session.Spec) -> dict:
    out = {}
    excluded = excluded_runs(root, spec)
    for m, ev in excluded.items():
        if m in session.CHRONO and ev == "preflight_failed":
            out[m] = {"pass": False, "preflight_failed": True}
    for model in [
        m for m in session.CHRONO + session.LADDER if (root / m).exists() and m not in excluded
    ]:
        t = session.model_table(model, spec)
        gt = t[t.block == "gate"].reset_index(drop=True)
        d = draws_for(root, model)
        wide = (
            d[d.draw >= 0]
            .pivot_table(index="prompt_id", columns="draw", values="parsed", dropna=False)
            .reindex(gt.prompt_id)
            .reindex(columns=range(K))
        )
        r = G.gate_stats(gt, wide.to_numpy(float))
        out[model] = {k: (v if not isinstance(v, np.bool_) else bool(v)) for k, v in r.items()}
    chrono = [m for m in session.CHRONO if m in out]
    passing = [m for m in chrono if out[m]["pass"]]
    priority = [f"chrono-{y}" for y in pairs.PRIORITY]
    n_pri = sum(m in passing for m in priority)
    return dict(
        per_model=out,
        passing_chrono=passing,
        priority_passing=n_pri,
        stop_rule_triggered=n_pri < 3,
        thresholds=G.THRESHOLDS,
    )


def substitutions_from(gate: dict) -> dict[str, str]:
    """Failed vintage -> latest earlier passing vintage (D9); vintages with none are dropped."""
    passing = [session.year(m) for m in gate["passing_chrono"]]
    subs = {}
    for m in session.CHRONO:
        if m in gate["per_model"] and m not in gate["passing_chrono"]:
            s = pairs.substitute(session.year(m) - 1, passing)
            if s is not None:
                subs[m] = f"chrono-{s}"
    return subs


# ---------------------------------------------------------------- Arm A
def arm_a_family(root: Path, spec: session.Spec, gate: dict) -> dict:
    models = gate["passing_chrono"]
    if not models:
        return dict(note="no ChronoGPT vintage passed the gate; Arm A is descriptive only")
    tables, draws = [], []
    for m in models:
        t = session.model_table(m, spec)
        tables.append(t[t.block == "main"])
        draws.append(draws_for(root, m))
    table = pd.concat(tables, ignore_index=True)
    res = A.analyse(table, pd.concat(draws, ignore_index=True), K)
    return dict(models=models, **_jsonable(res))


# ---------------------------------------------------------------- Arm B
def _armb_inputs(root: Path, spec: session.Spec, gate: dict):
    b = session.arm_b_table(spec.armb_variables, spec.armb_templates)
    draws = pd.concat(
        [draws_for(root, r) for r in runs_present(root) if r.startswith("chrono-")],
        ignore_index=True,
    ).drop_duplicates(["model", "prompt_id", "draw"])
    passing = [session.year(m) for m in gate["passing_chrono"]]
    rates = B.parse_rate_armb(b, draws)  # PAP section 5: Arm B also needs a 95% parse rate
    passing = [y for y in passing if rates.get(f"chrono-{y}", 0.0) >= G.THRESHOLDS["parse_rate"]]
    return b, draws, passing, rates


def arm_b_family(root: Path, spec: session.Spec, gate: dict, human: bool = True) -> dict:
    b, draws, passing, rates = _armb_inputs(root, spec, gate)
    run = lambda assign, **kw: B.build_draws_b(b, draws, assign, K, **kw)  # noqa: E731
    rolling = run(B.assignment("primary", passing=passing))
    out: dict = {"passing_years": passing, "armb_parse_rates": rates}
    out["H4"] = B.block_t(*rolling)
    contaminated = lambda v, t: pairs.contaminated(t, 2024)  # noqa: E731
    if 2024 in passing:
        c24 = run(B.assignment("fixed", 2024), pair_filter=contaminated)
        cvalid = run(B.assignment("primary", passing=passing), pair_filter=contaminated)
        out["H5"] = B.paired_contrast(c24, cvalid)
    else:
        out["H5"] = dict(note="2024 vintage failed the gate; H5 reported descriptively only")
    pv = {h: out[h]["p"] for h in ("H4", "H5") if "p" in out[h]}
    out["holm"] = dict(reject=holm(pv), adjusted=holm_adjusted(pv))

    # secondaries (outside the Holm family)
    sec: dict = {}
    if 2024 in passing:
        anon = dict(framing="anonymous", pair_filter=contaminated)
        a24 = run(B.assignment("fixed", 2024), **anon)
        avalid = run(B.assignment("primary", passing=passing), **anon)
        named_gap = B.block_estimates(*c24) - B.block_estimates(*cvalid)
        anon_gap = B.block_estimates(*a24) - B.block_estimates(*avalid)
        sec["framing_did"] = B.t_test(named_gap - anon_gap)
    if 1999 in passing and 2009 in passing:
        sec["vintage_2009_did"] = _did_2009(run)
    sec["per_variable"] = B.per_variable(*rolling).to_dict(orient="records")
    # human and benchmark comparisons on the same pairs (scaled units). Dry runs skip them so
    # no further SPF statistic on the Arm B sample is seen before the freeze.
    if not human:
        out["secondary"] = sec
        out["robustness"] = {}
        return _jsonable(out)
    cons, indiv = B.spf_draws(rolling[1], True), B.spf_draws(rolling[1], False)
    ar = B.ar_draws(rolling[1])
    sec["spf_consensus"] = B.block_t(*cons)
    sec["spf_individual"] = B.block_t(*indiv)
    sec["ar4_benchmark"] = B.block_t(*ar)
    sec["model_minus_spf_consensus"] = B.block_contrast_pairs(rolling, cons)
    single = B.block_t(*rolling, estimator=lambda d, cov: B.eor.naive_single(d, cov))
    sec["model_single_draw"] = single
    sec["model_single_minus_spf_individual"] = dict(
        b=single["b_full"] - sec["spf_individual"]["b_full"]
    )
    sec["rmse"] = dict(model=B.rmse(rolling[0]), spf_consensus=B.rmse(cons[0]), ar4=B.rmse(ar[0]))
    if 2024 in passing:
        sec["rmse_contaminated"] = dict(v2024=B.rmse(c24[0]), valid=B.rmse(cvalid[0]))
    out["secondary"] = sec

    # robustness rows
    rob = {}
    rob["latest_rule"] = B.block_t(*run(B.assignment("latest", passing=passing)))
    rob["six_vintage_rolling"] = B.block_t(*run(B.assignment("primary", six=True)))
    rob["drop_failed_quarters"] = B.block_t(
        *run(B.assignment("primary"), pair_filter=lambda v, t: pairs.assign(t)[0] in passing)
    )
    rob["covid_included"] = B.block_t(*run(B.assignment("primary", passing=passing), covid=False))
    rob["anonymous_framing"] = B.block_t(
        *run(B.assignment("primary", passing=passing), framing="anonymous")
    )
    rob["paraphrase_t"] = B.paraphrase_t(*rolling)
    if 1999 in passing:
        rob["constant_capability_1999"] = B.block_t(*run(B.assignment("fixed", 1999)))
        agree = B.paired_contrast(rolling, run(B.assignment("fixed", 1999)))
        rob["rolling_minus_1999"] = agree
    out["robustness"] = rob
    return _jsonable(out)


def _did_2009(run) -> dict:
    """[b(2009) - b(1999)] on pairs the 2009 vintage may have seen minus the same on pairs it
    could not have, by paraphrase (t-test across paraphrases)."""
    seen = lambda v, t: pairs.contaminated(t, 2009)  # noqa: E731
    unseen = lambda v, t: not pairs.contaminated(t, 2009) and pairs.assign(t)[0] >= 2009  # noqa: E731
    diffs = []
    parts = {}
    for name, flt in (("seen", seen), ("unseen", unseen)):
        d9, i9 = run(B.assignment("fixed", 2009), pair_filter=flt)
        d1, i1 = run(B.assignment("fixed", 1999), pair_filter=flt)
        parts[name] = (d9, i9, d1, i1)
    pars = sorted(set(parts["seen"][1].paraphrase))
    for p in pars:
        vals = []
        for name in ("seen", "unseen"):
            d9, i9, d1, i1 = parts[name]
            s9, s1 = (i9.paraphrase == p).to_numpy(), (i1.paraphrase == p).to_numpy()
            vals.append(B.pooled_b(d9.subset(s9)) - B.pooled_b(d1.subset(s1)))
        diffs.append(vals[0] - vals[1])
    return B.t_test(diffs)


# ---------------------------------------------------------------- output
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if hasattr(x, "b") and hasattr(x, "se") and hasattr(x, "psi"):  # Estimate
        return dict(
            b=x.b,
            se=x.se,
            ci=list(x.ci),
            p=x.p,
            n_units=x.n_units,
            first_stage_t2=x.first_stage_t2,
            fieller=x.fieller,
            fieller_kind=x.fieller_kind,
        )
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _md(family: str, res: dict) -> str:
    lines = [f"# Results: {family}", "", "Generated by `python -m ewh.confirm.run`.", ""]
    for h in ("H1", "H2", "H2_LS", "H3", "H3_LS", "H4", "H5"):
        if h in res and isinstance(res[h], dict) and "b" in res[h]:
            r = res[h]
            ci = r.get("ci", [float("nan")] * 2)
            lines.append(
                f"- **{h}**: {r['b']:.3f} (s.e. {r['se']:.3f}; 95% CI {ci[0]:.3f} to {ci[1]:.3f}; "
                f"p = {r['p']:.3g}" + (f"; MDE {r['mde']:.3f}" if "mde" in r else "") + ")"
            )
    if "holm" in res:
        lines.append(f"- Holm: {json.dumps(res['holm'])}")
    lines += ["", "Full results: the JSON file beside this one.", ""]
    return "\n".join(lines)


def _write(path_md: Path, path_json: Path, family: str, res: dict, once: bool) -> None:
    text, js = _md(family, res), json.dumps(res, indent=2, default=str)
    if once:
        binding.write_once(path_md, text)
        binding.write_once(path_json, js)
    else:
        path_md.parent.mkdir(parents=True, exist_ok=True)
        path_md.write_text(text)
        path_json.write_text(js)


# ---------------------------------------------------------------- dry generation
def dry_generate(root: Path = DRY_GEN) -> None:
    from ewh.models.backend import DryRunBackend
    from ewh.models.ledger import Ledger
    from ewh.models.runner import GenerationConfig, run_shards
    from ewh.sim.ar1 import seeds

    cfg = GenerationConfig(K, 1.0, 12, seeds()["dryrun"], session.ALLOC["shard_size"])
    for m in session.CHRONO:
        t = session.model_table(m, session.DRYRUN)
        run_shards(DryRunBackend(t, m), t, root, cfg, Ledger(root / "ledger.jsonl"), 500)
    binding.build_generation_manifest(root)


# ---------------------------------------------------------------- entry point
def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    family, dry = argv[0], "--dry-run" in argv
    if family == "dry-generate":
        dry_generate()
        return 0
    spec, root = (session.DRYRUN, DRY_GEN) if dry else (session.CONFIRM, GEN)
    prefix = "dryrun_" if dry else ""
    md, js = RESULTS / f"{prefix}{family}.md", RESULTS / f"{prefix}{family}.json"
    if not dry:
        binding.check_binding()
        gate_session = json.loads((root / "gate_session.json").read_text())
        excluded = excluded_runs(root, spec)
        expected = session.expected_shards(
            [m for m in session.CHRONO + session.LADDER if m not in excluded],
            gate_session.get("substitutions", {}),
        )
        binding.check_inputs(expected, root, RAW_ROOT)
        if family != "gate" and not (RESULTS / "gate.json").exists():
            raise binding.Refused("run the gate family first")
        binding.create_marker(family, RESULTS, binding.head())
    try:
        if family == "gate":
            res = gate_family(root, spec)
            res["substitutions"] = substitutions_from(res)
        else:
            gate = json.loads((RESULTS / f"{prefix}gate.json").read_text())
            if family == "arm_a":
                res = arm_a_family(root, spec, gate)
            else:
                res = arm_b_family(root, spec, gate, human=not dry)
    except Exception:
        if not dry:
            binding.write_once(RESULTS / f"{family}.failed.md", traceback.format_exc())
        raise
    _write(md, js, family, res, once=not dry)
    print(md.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
