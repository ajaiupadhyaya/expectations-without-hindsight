"""Arm B pipeline on the real pair structure with planted forecasters (two variables)."""

import numpy as np
import pandas as pd
import pytest

from ewh.analysis import arm_b
from ewh.data.philfed import SNAPSHOT
from ewh.prompts import tables as tb
from ewh.prompts.templates import ALL_TEMPLATES
from ewh.provenance.archive import RAW_ROOT
from ewh.sim import ar1

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "rtdsm" / SNAPSHOT / "ROUTPUTQvQd.xlsx").exists(), reason="no RTDSM snapshot"
)
K = 8


@pytest.fixture(scope="module")
def table():
    return tb.build_arm_b(variables=["RGDP", "UNEMP"], templates=ALL_TEMPLATES[:4])


def planted_draws(table, models, theta, noise, seed_tag):
    """Each model forecasts the target with a known coefficient theta on the revision:
    F_tm1 = w x_last_tm1 + (1-w) target (a noisy 'news' proxy), F_t = F_tm1 + lam (s - F_tm1)
    with s = target + independent news error; then draws add noise. Units keyed by unit_id."""
    rng = ar1.stream("tests", "armb", seed_tag)
    lam = 1.0 / (1.0 + theta)
    rows = []
    for model in models:
        for _uid, g in table.groupby("unit_id"):
            tm1 = g[g.date == "tm1"].iloc[0]
            t = g[g.date == "t"].iloc[0]
            y = t.target
            f_tm1 = 0.5 * tm1.x_last + 0.5 * y + rng.normal(0, 1.0)
            signal = y + rng.normal(0, 1.0)
            f_t = f_tm1 + lam * (signal - f_tm1)
            for row, f in ((tm1, f_tm1), (t, f_t)):
                vals = f + rng.normal(0, noise, K)
                for j, v in enumerate(vals):
                    rows.append(dict(model=model, prompt_id=row.prompt_id, draw=j, parsed=float(v)))
    return pd.DataFrame(rows)


def test_build_draws_and_block_t(table):
    draws = planted_draws(table, ["chrono-1999"], theta=0.0, noise=0.5, seed_tag="a")
    d, info = arm_b.build_draws_b(table, draws, arm_b.assignment("fixed", 1999), K)
    assert d.n == info.shape[0] > 0
    assert set(info.variable) == {"RGDP", "UNEMP"}
    # COVID targets excluded by default
    tq = info.t.map(pd.Period) + 3
    assert not ((tq >= arm_b.COVID[0]) & (tq <= arm_b.COVID[1])).any()
    r = arm_b.block_t(d, info)
    assert r["df"] == arm_b.BLOCKS - 1 and np.isfinite(r["se"])
    p = arm_b.paraphrase_t(d, info)
    assert p["df"] == 1  # 4 templates = paraphrases P1, P2 x 2 formats -> 2 groups
    pv = arm_b.per_variable(d, info)
    assert set(pv.variable) == {"RGDP", "UNEMP"}


def test_identical_runs_have_zero_contrast(table):
    draws = planted_draws(table, ["chrono-1999"], theta=0.0, noise=0.5, seed_tag="b")
    a = arm_b.build_draws_b(table, draws, arm_b.assignment("fixed", 1999), K)
    r = arm_b.paired_contrast(a, a)
    assert r["b_full"] == pytest.approx(0.0) and r["b"] == pytest.approx(0.0)


def test_primary_rule_uses_one_model_per_pair(table):
    vint = [f"chrono-{y}" for y in range(1999, 2025)]
    draws = planted_draws(table, vint, theta=0.0, noise=0.5, seed_tag="c")
    d, info = arm_b.build_draws_b(table, draws, arm_b.assignment("primary"), K)
    assert d.n > 0 and np.isfinite(d.f_t).mean() > 0.9


def test_blocks_are_consecutive():
    q = np.repeat(np.arange(100), 3)
    b = arm_b.blocks(q)
    assert set(b) == set(range(arm_b.BLOCKS))
    assert np.all(np.diff(b[np.argsort(q, kind="stable")]) >= 0)
