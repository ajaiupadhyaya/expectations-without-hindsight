"""The GPU session's allocation: which prompts each model runs, in which order (plan Phase 5).

Bound to tag pap-v1. `model_table(model)` is a deterministic function of the bound code, the
templates and the seeds, so the confirm scripts rebuild the exact tables and check their hashes.

Pass 1 (every model): preflight, then one table holding its gate, Arm A main block, crossed block
(1999 and 2024 only), its Arm B rows, and HellaSwag. Pass 2 (only if a vintage fails the gate):
the latest passing earlier vintage runs the failed vintage's rolling Arm B rows (D9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import pandas as pd

from ewh.data import pairs
from ewh.prompts import tables as tb
from ewh.prompts.templates import ALL_TEMPLATES

ALLOC = dict(
    k=32,  # sampled draws per prompt, plus one greedy read (Decision 10)
    temperature=1.0,
    max_new_tokens=12,
    arm_a_paths=40,  # per model x template x rho, all 26 vintages
    crossed_paths=20,  # per rho, shown under all 24 templates, 1999 and 2024 vintages
    crossed_models=("chrono-1999", "chrono-2024"),
    gate_n=100,  # per gate kind and rho
    ladder_paths=6,  # per template x rho for each ladder model (descriptive)
    shard_size=2000,
    armb_full=(1999, 2009, 2024),
)
CHRONO = tuple(f"chrono-{y}" for y in range(1999, 2025))
LADDER = (
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
)


def year(model: str) -> int:
    return int(model.split("-")[1])


@cache
def arm_b_table(variables: tuple[str, ...] | None = None, n_templates: int | None = None):
    tpl = ALL_TEMPLATES if n_templates is None else ALL_TEMPLATES[:n_templates]
    return tb.build_arm_b(variables=list(variables) if variables else None, templates=tpl)


def _arm_b_needed(y: int, t: pd.Period, date: str) -> bool:
    """Is (pair t, date) needed from vintage y by any pre-registered Arm B run?"""
    if y in ALLOC["armb_full"]:
        return True  # 1999, 2009, 2024 run every quarter
    p_tm1, p_t = pairs.assign(t, "primary")
    if p_tm1 == y:
        return True  # annual rolling, primary rule (both dates)
    l_tm1, l_t = pairs.assign(t, "latest")
    if (date == "tm1" and l_tm1 == y) or (date == "t" and l_t == y):
        return True  # latest-vintage robustness rule
    return pairs.six_vintage_fallback(p_tm1) == y  # six-vintage fallback rolling


def arm_b_rows(model: str, b: pd.DataFrame | None = None) -> pd.DataFrame:
    y = year(model)
    b = arm_b_table() if b is None else b
    keep = [_arm_b_needed(y, pd.Period(t), d) for t, d in zip(b.t, b.date, strict=True)]
    return b[keep]


def substitution_rows(failed: str, b: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pass 2: the failed vintage's annual-rolling rows (primary rule), run by its substitute."""
    y = year(failed)
    b = arm_b_table() if b is None else b
    keep = [pairs.assign(pd.Period(t), "primary")[0] == y for t in b.t]
    return b[keep]


@dataclass(frozen=True)
class Spec:
    """Namespaces and sizes. The default is the confirmatory session; dry runs pass small sizes
    and the `dryrun` namespace so no confirmatory seed is touched."""

    ns_gate: str = "gate"
    ns_main: str = "confirm"
    arm_a_paths: int = ALLOC["arm_a_paths"]
    crossed_paths: int = ALLOC["crossed_paths"]
    gate_n: int = ALLOC["gate_n"]
    ladder_paths: int = ALLOC["ladder_paths"]
    armb_variables: tuple[str, ...] | None = None
    armb_templates: int | None = None


CONFIRM = Spec()
DRYRUN = Spec("dryrun", "dryrun", 4, 2, 12, 2, ("RGDP", "UNEMP"), 4)


@cache
def model_table(model: str, spec: Spec = CONFIRM) -> pd.DataFrame:
    parts = [tb.build_gate(spec.ns_gate, [model], spec.gate_n)]
    if model in CHRONO:
        parts.append(tb.build_arm_a(spec.ns_main, [model], spec.arm_a_paths))
        if model in ALLOC["crossed_models"]:
            parts.append(tb.build_crossed(spec.ns_main, [model], spec.crossed_paths))
        parts.append(arm_b_rows(model, arm_b_table(spec.armb_variables, spec.armb_templates)))
    else:
        parts.append(tb.build_arm_a(spec.ns_main, [model], spec.ladder_paths))
    t = pd.concat(parts, ignore_index=True)
    return t.drop_duplicates("prompt_id").reset_index(drop=True)


def n_shards(table: pd.DataFrame) -> int:
    return math.ceil(len(table) / ALLOC["shard_size"])


def expected_shards(models, substitutions=None, spec: Spec = CONFIRM) -> dict[str, int]:
    """Shard counts per run name; substitution runs are named `<substitute>-for-<failed>`."""
    out = {m: n_shards(model_table(m, spec)) for m in models}
    b = arm_b_table(spec.armb_variables, spec.armb_templates)
    for failed, sub in (substitutions or {}).items():
        out[f"{sub}-for-{failed}"] = n_shards(substitution_rows(failed, b))
    return out
