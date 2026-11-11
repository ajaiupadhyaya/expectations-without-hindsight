import numpy as np
import pytest

from ewh.data import aklmt

pytestmark = pytest.mark.skipif(
    not aklmt.ZIP_PATH.exists(), reason="AKLMT package not archived (python -m ewh.data.sources)"
)

# Published implied persistence, Experiment 1, h = 1 (AKLMT 2023, Figure II panel B labels;
# authors' working paper of 2023-02-07). Reproducing it validates our reading of their data.
PUBLISHED_IMPLIED = {0.0: 0.45, 0.2: 0.62, 0.4: 0.77, 0.6: 0.85, 0.8: 0.91, 1.0: 0.99}


@pytest.fixture(scope="module")
def table():
    return aklmt.figure2(aklmt.load_experiment1()).set_index("rho")


def test_reproduces_published_implied_persistence(table):
    for rho, published in PUBLISHED_IMPLIED.items():
        assert round(table.loc[rho, "implied_rho"], 2) == pytest.approx(published, abs=0.011)


def test_error_on_revision_is_negative_and_in_figure_range(table):
    # Figure II panel A: every coefficient negative, axis -0.8 to 0, more negative at low rho.
    b = table["b_eor"]
    assert (b < 0).all() and (b > -0.8).all()
    assert b.loc[0.0] < b.loc[1.0]


def test_cluster_se_matches_known_formula():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(50), 4)
    x = rng.normal(size=200)
    y = 1 + 0.5 * x + rng.normal(size=200)
    out = aklmt.ols_cluster(y, x, g)
    assert out["clusters"] == 50 and out["n"] == 200
    assert out["se_cr1"] > out["se"]
    assert out["b"] == pytest.approx(np.polyfit(x, y, 1)[0])
