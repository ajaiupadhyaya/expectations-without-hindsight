"""CPU pilot on a non-confirmatory model (plan Task 3.7 calibration; logged in docs/pilot_log.md).

Qwen/Qwen2-0.5B-Instruct (parsing stand-in, not in the ladder), pilot seeds, rho = 0.5 only, all
12 templates. Measures parse rate, noise share, prompts per second on this CPU, and the spread of
the corrected coefficient across paraphrases (tau), which drives the template-inference MDE.
`python -m ewh.pilot [n_paths]`
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from ewh.analysis import arm_a
from ewh.estimators import eor
from ewh.estimators.persistence import noise_share
from ewh.models.ledger import Ledger
from ewh.models.runner import GenerationConfig, read_draws, run_shards
from ewh.prompts import tables as tb
from ewh.provenance.archive import REPO_ROOT
from ewh.sim import ar1

OUT = REPO_ROOT / "data" / "pilot" / "qwen2-0.5b"
K = 16


def main(argv):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ewh.models.backend import HFBackend

    n_paths = int(argv[0]) if argv else 40
    m = yaml.safe_load((REPO_ROOT / "configs" / "generation.yaml").read_text())["models"][
        "Qwen2-0.5B-Instruct"
    ]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        m["repo"], revision=m["revision"], torch_dtype=torch.float32
    ).eval()
    torch.set_num_threads(4)
    be = HFBackend(model, tok, m["revision"], "qwen2-0.5b", chat=True)
    table = tb.build_arm_a("pilot", ["qwen2-0.5b"], n_paths, rhos=(0.5,))
    seed = ar1.seeds()["pilot"]
    cfg = GenerationConfig(k=K, temperature=1.0, max_new_tokens=12, master_seed=seed, shard_size=96)
    t0 = time.time()
    run_shards(be, table, OUT / "draws", cfg, Ledger(OUT / "ledger.jsonl"), batch_size=8)
    secs = time.time() - t0
    draws = read_draws(OUT / "draws", "qwen2-0.5b")
    d = arm_a.build_draws(table, draws, K)
    valid = np.isfinite(d.f_t).mean() * 0.5 + np.isfinite(d.f_tm1).mean() * 0.5
    cov = eor.CovSpec("cr1", ("path",)) if hasattr(eor, "CovSpec") else None
    from ewh.estimators.types import CovSpec

    cov = CovSpec("cr1", ("path",))
    units = table.drop_duplicates("unit_id")
    par = units.paraphrase.to_numpy()
    per = {}
    for p in sorted(set(par)):
        sub = d.subset(par == p)
        e = eor.split_iv(sub, cov)
        per[p] = (e.b, e.se, e.n_units)
    bs = np.array([v[0] for v in per.values()])
    ses = np.array([v[1] for v in per.values()])
    tau2 = max(0.0, bs.var(ddof=1) - np.mean(ses**2))
    allp = eor.split_iv(d, cov)
    naive = eor.naive_stacked(d, cov)
    res = dict(
        model=m["repo"],
        revision=m["revision"],
        n_prompts=len(table),
        k=K,
        seconds=secs,
        prompts_per_second=len(table) / secs if secs > 0 else None,
        valid_share=float(valid),
        units_in_sample=allp.n_units,
        noise_share_t=noise_share(d),
        noise_share_tm1=noise_share(d, "tm1"),
        split_iv=allp.b,
        split_iv_se=allp.se,
        naive_stacked=naive.b,
        naive_se=naive.se,
        per_paraphrase={k: dict(b=v[0], se=v[1], n=v[2]) for k, v in per.items()},
        tau_hat=float(np.sqrt(tau2)),
        between_paraphrase_sd=float(bs.std(ddof=1)),
        examples=draws[draws.draw >= 0].raw_text.head(8).tolist(),
    )
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
