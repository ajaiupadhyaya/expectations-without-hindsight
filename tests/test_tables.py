import numpy as np
import pandas as pd
import pytest
import tiktoken

from ewh.prompts import tables as tb
from ewh.prompts.templates import ALL_TEMPLATES, chronogpt_wrap, templates_sha256

ENC = tiktoken.get_encoding("gpt2")


def _numbers(text):
    return [float(x) for x in text.split("\n")[1].split(", ")]


def test_templates_hash_is_recorded():
    import yaml

    from ewh.provenance.archive import REPO_ROOT

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "generation.yaml").read_text())
    assert cfg["templates_sha256"] == templates_sha256()
    assert len(ALL_TEMPLATES) == 24


def test_chronogpt_wrapper_matches_vendored_code():
    from ewh.provenance.archive import REPO_ROOT

    src = (REPO_ROOT / "vendor" / "chronogpt" / "ChronoGPT_instruct.py").read_text()
    wrapped = chronogpt_wrap("INPUT", "")
    assert "You are ChronoGPT" in wrapped
    assert (
        '"""You are ChronoGPT, a large language model trained by ManelaLab at WashU.\n    Below'
        in src
    )
    assert wrapped.endswith("\nINPUT\n\n### Input:\n### Response:\n")


def test_tables_rebuild_byte_identical():
    a1 = tb.build_arm_a("dryrun", ["m"], 3)
    a2 = tb.build_arm_a("dryrun", ["m"], 3)
    assert tb.table_sha256(a1) == tb.table_sha256(a2)
    assert tb.table_sha256(a1) != tb.table_sha256(tb.build_arm_a("pilot", ["m"], 3))


def test_pair_structure_and_targets():
    a = tb.build_arm_a("dryrun", ["m"], 2, rhos=(0.6,))
    for _, g in a.groupby("unit_id"):
        tm1, t = g[g.date == "tm1"].iloc[0], g[g.date == "t"].iloc[0]
        assert tm1.h == 2 and t.h == 1 and tm1.target == t.target
        if tm1.fmt == "comma":
            x_tm1, x_t = _numbers(tm1.input_text), _numbers(t.input_text)
            assert len(x_tm1) == 40 and len(x_t) == 41 and x_t[:40] == x_tm1
            assert t.x_last == x_t[-1]


def test_nested_block_uses_fresh_paths_crossed_block_shares():
    a = tb.build_arm_a("dryrun", ["m1", "m2"], 2, rhos=(0.6,))
    texts = (
        a[(a.date == "tm1") & (a.fmt == "comma")].groupby(["model", "template"]).input_text.first()
    )
    assert texts.map(lambda s: tuple(_numbers(s))).nunique() == len(texts)
    c = tb.build_crossed("dryrun", ["m1", "m2"], 1, rhos=(0.6,))
    c = c[(c.date == "tm1") & (c.fmt == "comma")]
    assert c.input_text.map(lambda s: tuple(_numbers(s))).nunique() == 1
    assert c.path_uid.nunique() == 1


def test_gate_pairs():
    g = tb.build_gate("dryrun", ["m"], 3)
    a = g[g.kind == "hist_a"].set_index("pair")
    b = g[g.kind == "hist_b"].set_index("pair")
    assert (a.x_last == b.x_last).all() and (a["hist"] != b["hist"]).all()
    o = g[g.kind == "scale_orig"].set_index("pair")
    s = g[g.kind == "scale_trans"].set_index("pair")
    assert np.allclose(s.x_last, 3 * o.x_last + 500)
    assert set(g[g.kind == "persistence"].rho) == {0.0, 0.5, 1.0}


@pytest.fixture(scope="module")
def arm_b():
    from ewh.data.philfed import SNAPSHOT
    from ewh.provenance.archive import RAW_ROOT

    if not (RAW_ROOT / "rtdsm" / SNAPSHOT / "ROUTPUTQvQd.xlsx").exists():
        pytest.skip("RTDSM snapshot not archived")
    return tb.build_arm_b(variables=["RGDP", "UNEMP"])


def test_arm_b_prompts_use_only_the_survey_vintage(arm_b):
    from ewh.data import realtime as rt

    row = arm_b[
        (arm_b.variable == "RGDP") & (arm_b.fmt == "comma") & (arm_b.framing == "anonymous")
    ]
    r = row.iloc[0]
    s = pd.Period(r.t) - (1 if r.date == "tm1" else 0)
    hist = rt.history("RGDP", s, 40)
    assert _numbers(r.input_text) == [float(f"{v:.1f}") for v in hist]
    assert all(
        pd.Period(q) < pd.Period(t) for q, t in zip(arm_b.last_obs, arm_b.t, strict=True) if True
    )
    # the target quarter is never among the shown quarters
    named = arm_b[arm_b.framing == "named"].iloc[0]
    assert str(pd.Period(named.t) + 3) not in named.input_text.split("\n")[1]


def test_no_prompt_exceeds_chronogpt_context(arm_b):
    a = tb.build_arm_a("dryrun", ["m"], 2)
    for df in (a, arm_b):
        n = max(
            len(ENC.encode(chronogpt_wrap(t, p)))
            for t, p in zip(df.input_text, df.prefill, strict=True)
        )
        assert n < 1792 - 16
