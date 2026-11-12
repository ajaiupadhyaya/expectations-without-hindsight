import numpy as np
import pytest

from ewh.capability import hellaswag as hs

pytestmark = pytest.mark.skipif(not hs.PATH.exists(), reason="HellaSwag not archived")


def test_subset_is_seeded_and_complete():
    a, b = hs.items(), hs.items()
    assert len(a) == 1000 and a.ind.tolist() == b.ind.tolist()
    assert all(len(e) == 4 for e in a.endings) and set(a.label) <= {0, 1, 2, 3}
    assert not any("[" in q for q in a["query"])


def test_oracle_and_chance_scorers():
    it = hs.items(200)
    labels = np.repeat(it.label.to_numpy(), 4)
    endings_idx = np.tile(np.arange(4), len(it))

    def oracle(ctxs, conts):
        return np.where(endings_idx == labels, 0.0, -10.0)

    assert hs.score(it, oracle)["acc"] == 1.0
    rng = np.random.default_rng(0)
    r = hs.score(it, lambda c, e: rng.normal(size=len(c)))
    assert 0.15 < r["acc"] < 0.35
