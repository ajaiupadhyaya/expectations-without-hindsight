"""AR(1) paths in the AKLMT (2023) design: x_{t+1} = rho x_t + e, mean 0, sd(e) = 20.

Every stream is derived from (namespace seed, tags) with numpy SeedSequence, so different rho
values, namespaces or purposes never share shocks (plan D2).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import yaml

SEEDS_PATH = Path(__file__).resolve().parents[2] / "configs" / "seeds.yaml"
SD = 20.0
RHOS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@cache
def seeds() -> dict[str, int]:
    return yaml.safe_load(SEEDS_PATH.read_text())


def stream(namespace: str | int, *tags: int | str) -> np.random.Generator:
    """Independent generator for (namespace, tags). Strings are hashed stably."""
    root = seeds()[namespace] if isinstance(namespace, str) else int(namespace)
    key = tuple(_tag(t) for t in tags)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(root, spawn_key=key)))


def _tag(t: int | str) -> int:
    if isinstance(t, int):
        return t
    return int.from_bytes(t.encode(), "little") % (2**63)


def rho_tag(rho: float) -> int:
    return int(round(rho * 1000))


def ar1_paths(
    rho: float,
    n_paths: int,
    length: int,
    seed: str | int,
    *tags: int | str,
    sd: float = SD,
    integer: bool = False,
) -> np.ndarray:
    """(n_paths, length) array. Stationary start for |rho| < 1; a random walk starts at 0.

    `integer=True` rounds each realization, as AKLMT show integers to participants. The shocks
    are drawn from a stream keyed on (seed, rho, *tags).
    """
    rng = stream(seed, rho_tag(rho), *tags)
    eps = rng.normal(0.0, sd, size=(n_paths, length))
    x = np.empty_like(eps)
    x[:, 0] = eps[:, 0] / np.sqrt(1 - rho**2) if abs(rho) < 1 else 0.0
    for s in range(1, length):
        x[:, s] = rho * x[:, s - 1] + eps[:, s]
    return np.rint(x) if integer else x
