import numpy as np
import pytest

from ewh.capability import gate
from ewh.prompts import tables as tb
from ewh.sim import ar1


@pytest.fixture(scope="module")
def table():
    return tb.build_gate("tests", ["m"], 150)


@pytest.mark.parametrize("kind", ["ls", "overreactor"])
def test_gate_passes_competent_forecasters(table, kind):
    d = gate.synthetic(table, kind, 8, ar1.stream("tests", "gate", kind))
    r = gate.gate_stats(table, d)
    assert r["pass"], r


@pytest.mark.parametrize("kind", ["echo", "noisy_echo", "constant"])
def test_gate_fails_weak_forecasters(table, kind):
    d = gate.synthetic(table, kind, 8, ar1.stream("tests", "gate", kind))
    r = gate.gate_stats(table, d)
    assert not r["pass"], r
    assert not r["pass_persistence"] or not r["pass_history"]


def test_parse_rate_counts_invalid_draws(table):
    d = gate.synthetic(table, "ls", 8, ar1.stream("tests", "gate", "parse"))
    d[:, :1] = np.nan  # one of eight draws unparsed everywhere
    r = gate.gate_stats(table, d)
    assert r["parse_rate"] == pytest.approx(7 / 8) and not r["pass_parse"]


def test_overreactor_implied_persistence_matches_aklmt(table):
    d = gate.synthetic(table, "overreactor", 8, ar1.stream("tests", "gate", "aklmt"))
    r = gate.gate_stats(table, d)
    assert r["implied_rho"][0.0] == pytest.approx(0.45, abs=0.08)
    assert r["implied_rho"][1.0] == pytest.approx(0.99, abs=0.08)
