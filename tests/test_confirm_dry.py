"""The three confirm families run end to end on a synthetic session (dry-run spec)."""

import json

import pytest

from ewh.confirm import run as R
from ewh.data.philfed import SNAPSHOT
from ewh.provenance.archive import RAW_ROOT

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (RAW_ROOT / "rtdsm" / SNAPSHOT).exists(), reason="no snapshot"),
]


def test_dry_confirm_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DRY_GEN", tmp_path / "gen")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    R.dry_generate(tmp_path / "gen")
    for fam in ("gate", "arm_a", "arm_b"):
        assert R.main([fam, "--dry-run"]) == 0
    a = json.loads((tmp_path / "results" / "dryrun_arm_a.json").read_text())
    assert a["H2"]["b"] == pytest.approx(-0.3, abs=0.05)  # planted coefficient recovered
    assert a["H2"]["df"] == 11
    b = json.loads((tmp_path / "results" / "dryrun_arm_b.json").read_text())
    assert b["H4"]["n_nonfinite"] == 0 and "p" in b["H5"]
