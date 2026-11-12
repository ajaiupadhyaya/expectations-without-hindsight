"""Prompt tables for Arm A (main and crossed blocks), the competence gate, and Arm B.

Every row is one prompt. Rows sharing a `unit_id` form a revision pair (date "tm1" and "t").
`path_uid` is the clustering key (plan D2): the realized shock is shared by every prompt that
shows the same path. Tables are deterministic functions of (namespace seed, sizes, templates).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ewh.prompts.templates import ALL_TEMPLATES, Template, prompt_id, render, spec
from ewh.sim.ar1 import RHOS, ar1_paths

SHOWN = 40  # values shown at t-1; t shows SHOWN + 1
GATE_RHOS = (0.0, 0.5, 1.0)


def _row(**kw) -> dict:
    kw["prompt_id"] = prompt_id(kw["input_text"], kw["prefill"])
    return kw


def _pair_rows(base: dict, path: np.ndarray, template: Template) -> list[dict]:
    """Date t-1 sees x[:SHOWN] and forecasts x[SHOWN+1] two steps ahead; date t sees
    x[:SHOWN+1] and forecasts the same target one step ahead."""
    rows = []
    for date, shown, h in (("tm1", SHOWN, 2), ("t", SHOWN + 1, 1)):
        text, prefill = render(template, list(path[:shown]), h)
        rows.append(
            _row(
                **base,
                date=date,
                h=h,
                input_text=text,
                prefill=prefill,
                target=float(path[SHOWN + 1]),
                x_last=float(path[shown - 1]),
                hist=",".join(str(int(v)) for v in path[:shown]),
            )
        )
    return rows


def build_arm_a(
    namespace: str,
    models: list[str],
    n_paths: int,
    rhos: tuple[float, ...] = RHOS,
    templates: tuple[Template, ...] = ALL_TEMPLATES,
) -> pd.DataFrame:
    """Main block: a fresh set of n_paths paths for every model x template x rho (nested)."""
    rows = []
    for m in models:
        for tpl in templates:
            for rho in rhos:
                paths = ar1_paths(
                    rho, n_paths, SHOWN + 2, namespace, "armA", "main", m, tpl.name, integer=True
                )
                for p in range(n_paths):
                    puid = f"main|{m}|{tpl.name}|{rho}|{p}"
                    base = dict(
                        block="main",
                        model=m,
                        paraphrase=tpl.paraphrase,
                        fmt=tpl.fmt,
                        template=tpl.name,
                        rho=rho,
                        path_uid=puid,
                        unit_id=puid,
                    )
                    rows += _pair_rows(base, paths[p], tpl)
    return pd.DataFrame(rows)


def build_crossed(
    namespace: str,
    models: list[str],
    n_paths: int,
    rhos: tuple[float, ...] = RHOS,
    templates: tuple[Template, ...] = ALL_TEMPLATES,
) -> pd.DataFrame:
    """Crossed block: the same n_paths paths per rho shown under every template and to every
    listed model (for the cross-template instrument)."""
    rows = []
    for rho in rhos:
        paths = ar1_paths(rho, n_paths, SHOWN + 2, namespace, "armA", "crossed", integer=True)
        for p in range(n_paths):
            puid = f"crossed|{rho}|{p}"
            for m in models:
                for tpl in templates:
                    base = dict(
                        block="crossed",
                        model=m,
                        paraphrase=tpl.paraphrase,
                        fmt=tpl.fmt,
                        template=tpl.name,
                        rho=rho,
                        path_uid=puid,
                        unit_id=f"{puid}|{m}|{tpl.name}",
                    )
                    rows += _pair_rows(base, paths[p], tpl)
    return pd.DataFrame(rows)


def build_gate(namespace: str, models: list[str], n: int) -> pd.DataFrame:
    """Competence-gate prompts (plan D9), one-step forecasts only, templates in rotation.

    kinds: "persistence" (rho in GATE_RHOS); "hist_a"/"hist_b" pairs of histories with the same
    last value, from rho 0.2 and rho 0.8 (b shifted so its last value equals a's);
    "scale_orig"/"scale_trans" (x and 3x + 500)."""
    rows = []
    T = len(ALL_TEMPLATES)
    for m in models:
        k = 0

        def add(kind, pair, path, rho, m=m):
            nonlocal k
            tpl = ALL_TEMPLATES[k % T]
            k += 1
            text, prefill = render(tpl, list(path[:SHOWN]), 1)
            rows.append(
                _row(
                    block="gate",
                    model=m,
                    kind=kind,
                    pair=pair,
                    rho=rho,
                    paraphrase=tpl.paraphrase,
                    fmt=tpl.fmt,
                    template=tpl.name,
                    path_uid=f"gate|{m}|{kind}|{pair}",
                    input_text=text,
                    prefill=prefill,
                    x_last=float(path[SHOWN - 1]),
                    target=float(path[SHOWN]),
                    hist=",".join(str(int(v)) for v in path[:SHOWN]),
                )
            )

        for rho in GATE_RHOS:
            for i, path in enumerate(
                ar1_paths(rho, n, SHOWN + 1, namespace, "gate", m, integer=True)
            ):
                add("persistence", f"{rho}|{i}", path, rho)
        a = ar1_paths(0.2, n, SHOWN + 1, namespace, "gate-hist-a", m, integer=True)
        b = ar1_paths(0.8, n, SHOWN + 1, namespace, "gate-hist-b", m, integer=True)
        for i in range(n):
            add("hist_a", str(i), a[i], 0.2)
            add("hist_b", str(i), b[i] - b[i, SHOWN - 1] + a[i, SHOWN - 1], 0.8)
        s = ar1_paths(0.5, n, SHOWN + 1, namespace, "gate-scale", m, integer=True)
        for i in range(n):
            add("scale_orig", str(i), s[i], 0.5)
            add("scale_trans", str(i), 3 * s[i] + 500, 0.5)
    return pd.DataFrame(rows)


def build_arm_b(variables=None, framings=("named", "anonymous"), templates=ALL_TEMPLATES):
    """Arm B prompts (Decision 7). Not model-specific: model assignment is in ewh.data.pairs.
    Pairs whose survey-quarter vintage is stale are dropped (OQ2, proposed default (a))."""
    from ewh.data import pairs
    from ewh.data import realtime as rt

    variables = variables or rt.VARIABLES
    series = spec()["series"]
    rows = []
    for var in variables:
        for t in pairs.forecast_quarters(var):
            try:
                hists = {"tm1": rt.history(var, t - 1, SHOWN), "t": rt.history(var, t, SHOWN)}
            except ValueError:
                continue  # stale vintage (OQ2)
            target = rt.outcome(var, t)
            if not np.isfinite(target):
                continue  # outcome never published in the snapshot (e.g. UNEMP 2025Q4)
            for framing in framings:
                for tpl in templates:
                    uid = f"armB|{var}|{t}|{framing}|{tpl.name}"
                    for date, h in (("tm1", 5), ("t", 4)):
                        hist = hists[date]
                        if framing == "named":
                            labels = [str(q) for q in hist.index]
                            target_label = str(t + 3)
                            named = dict(
                                series=series[var],
                                first=labels[0],
                                last=labels[-1],
                                target_q=target_label,
                            )
                        else:
                            labels, target_label, named = None, None, None
                        text, prefill = render(
                            tpl,
                            list(hist.to_numpy()),
                            h,
                            decimals=1,
                            labels=labels,
                            target_label=target_label,
                            named=named,
                        )
                        rows.append(
                            _row(
                                block="armB",
                                variable=var,
                                t=str(t),
                                date=date,
                                h=h,
                                framing=framing,
                                paraphrase=tpl.paraphrase,
                                fmt=tpl.fmt,
                                template=tpl.name,
                                unit_id=uid,
                                path_uid=str(t),
                                input_text=text,
                                prefill=prefill,
                                target=target,
                                x_last=float(hist.iloc[-1]),
                                last_obs=str(hist.index[-1]),
                                hist=",".join(f"{v:.1f}" for v in hist.to_numpy()),
                            )
                        )
    return pd.DataFrame(rows)


def table_sha256(df: pd.DataFrame) -> str:
    cols = sorted(df.columns)
    payload = df[cols].to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()
