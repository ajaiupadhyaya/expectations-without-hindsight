"""Deviation 1 (2026-11-12): runs the session records as not generated leave the expected shard set;
a preflight failure counts as failing the gate."""

from ewh import session
from ewh.confirm import run as R
from ewh.models.ledger import Ledger

LADDER_M = "Llama-3.2-1B-Instruct"


def test_skipped_and_preflight_failed_runs_are_excluded(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append(event="preflight_failed", model="chrono-1999", top1_agree=0.5, passed=False)
    led.append(event="skipped", model=LADDER_M, reason="GatedRepoError")
    assert R.excluded_runs(tmp_path, session.DRYRUN) == {
        "chrono-1999": "preflight_failed",
        LADDER_M: "skipped",
    }


def test_a_skipped_model_that_later_completed_is_not_excluded(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append(event="skipped", model=LADDER_M, reason="OSError")
    n = session.n_shards(session.model_table(LADDER_M, session.DRYRUN))
    for s in range(n):
        led.append(event="done", model=LADDER_M, shard=s, sha256=f"h{s}")
    assert R.excluded_runs(tmp_path, session.DRYRUN) == {}


def test_no_ledger_means_nothing_excluded(tmp_path):
    assert R.excluded_runs(tmp_path, session.DRYRUN) == {}


def test_preflight_failure_fails_the_gate_and_counts_for_the_stop_rule(tmp_path):
    Ledger(tmp_path / "ledger.jsonl").append(event="preflight_failed", model="chrono-1999")
    g = R.gate_family(tmp_path, session.DRYRUN)
    assert g["per_model"]["chrono-1999"] == {"pass": False, "preflight_failed": True}
    assert "chrono-1999" not in g["passing_chrono"]
    assert g["priority_passing"] == 0 and g["stop_rule_triggered"]


def test_preflight_failure_is_substituted_like_a_gate_failure():
    gate = {
        "per_model": {
            "chrono-1999": {"pass": True},
            "chrono-2004": {"pass": False, "preflight_failed": True},
        },
        "passing_chrono": ["chrono-1999"],
    }
    assert R.substitutions_from(gate) == {"chrono-2004": "chrono-1999"}
