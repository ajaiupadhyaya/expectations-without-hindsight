import numpy as np
import pandas as pd
import pytest

from ewh.data import spf
from ewh.data.philfed import SNAPSHOT
from ewh.provenance.archive import RAW_ROOT

HAVE_DATA = (RAW_ROOT / "spf" / SNAPSHOT / "SPFmicrodata.xlsx").exists()
HAVE_BGMS = (RAW_ROOT / "bgms" / "118422-V1.zip").exists()


def test_stata_percentile_definition():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spf.stata_pctile(x, 50) == 2.5  # n*p = 2 is an integer: average of 2nd and 3rd
    assert spf.stata_pctile(x, 25) == 1.5
    assert spf.stata_pctile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 25) == 2.0  # ceil(1.25) = 2nd
    assert np.isnan(spf.stata_pctile(np.array([np.nan]), 50))


def test_trim_removes_only_outliers():
    df = pd.DataFrame(
        {
            "yq": [pd.Period("2001Q1")] * 6,
            "ID": range(6),
            **{f"c{k}": [1.0, 1.1, 0.9, 1.05, 0.95, 1.0] for k in range(1, 7)},
        }
    )
    df.loc[5, "c5"] = 50.0
    out = spf.trim(df)
    assert np.isnan(out.loc[5, "c5"]) and out.c5.notna().sum() == 5
    assert out.c1.notna().all()  # column 1 is never trimmed (BGMS trim columns 2-6)


def test_sample_drops_forecasters_with_few_pairs():
    t = pd.period_range("2001Q1", periods=12, freq="Q")
    df = pd.DataFrame(
        {
            "ID": [1] * 12 + [2] * 5,
            "t": list(t) + list(t[:5]),
            "err": 1.0,
            "rev": 0.5,
        }
    )
    s = spf.sample(df, t[0], t[-1])
    assert s[s.ID == 1].err.notna().all() and s[s.ID == 2].err.isna().all()


@pytest.mark.skipif(not (HAVE_DATA and HAVE_BGMS), reason="needs SPF snapshot and BGMS package")
def test_replication_check_passes():
    from ewh.data import spf_replication as rep

    ref, mine, cg = rep.bgms_reference(), rep.ours(), rep.cg_check()
    ok, _ = rep.verdict(ref, mine, cg)
    assert ok
    for v in rep.HEADLINE:  # BGMS's own panel reproduces the published Table 3
        assert ref[v]["cons_b"] == pytest.approx(rep.PUBLISHED_2019[v][0], abs=0.005)
        assert ref[v]["ind_b"] == pytest.approx(rep.PUBLISHED_2019[v][1], abs=0.005)
