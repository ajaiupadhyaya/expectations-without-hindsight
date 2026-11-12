"""The confirmatory GPU session (plan Phase 5). Run ON THE POD from a checkout of tag pap-v1:

    uv run python pod/run_session.py            # resumable; safe to rerun after a crash

Order: the six priority vintages (1999, 2004, 2009, 2014, 2019, 2024), then the gate and the stop
rule (fewer than 3 of 6 pass -> write STOP and exit for AJ), then the other 20 vintages, then pass 2
(substitution runs for vintages that failed the gate), then the ladder models smallest first.
Per model: pinned download, preflight (ChronoGPT: Decision 11), every shard of
`ewh.session.model_table(model)`, HellaSwag, delete the weights. Outputs go to data/generations/
(raw draws, ledger, capability scores, preflight results, gate_session.json, session_env.json).

Allowed in-session diagnostics (PAP): row counts, parse rate, tokens per second, preflight, and
the gate (needed for the stop rule and substitutions). Nothing else is computed here.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ewh import session  # noqa: E402
from ewh.capability import hellaswag as hs  # noqa: E402
from ewh.confirm import binding  # noqa: E402
from ewh.confirm import run as R  # noqa: E402
from ewh.models import chrono_fast as cf  # noqa: E402
from ewh.models.backend import ChronoBackend, HFBackend  # noqa: E402
from ewh.models.ledger import Ledger  # noqa: E402
from ewh.models.runner import GenerationConfig, run_shards  # noqa: E402
from ewh.prompts import tables as tb  # noqa: E402
from ewh.prompts.templates import chronogpt_wrap  # noqa: E402
from ewh.sim.ar1 import seeds  # noqa: E402

GEN = ROOT / "data" / "generations"
CFG = yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text())
PRIORITY = [f"chrono-{y}" for y in (1999, 2004, 2009, 2014, 2019, 2024)]
OTHERS = [m for m in session.CHRONO if m not in PRIORITY]
BATCH = 32


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def gen_config() -> GenerationConfig:
    a = session.ALLOC
    return GenerationConfig(
        a["k"], a["temperature"], a["max_new_tokens"], seeds()["confirm"], a["shard_size"]
    )


def load_chrono(model: str):
    from huggingface_hub import hf_hub_download

    m = CFG["models"][model]
    raw = torch.load(
        hf_hub_download(m["repo"], "config.pt", revision=m["revision"]),
        map_location="cpu",
        weights_only=True,
    )
    config = {k: raw[k] for k in ("vocab_size", "num_layers", "num_heads", "model_dim")}
    path = hf_hub_download(m["repo"], "pytorch_model.bin", revision=m["revision"])
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    return sd, config, m["revision"], Path(path)


def preflight(sd, config, fast) -> dict:
    """Decision 11: teacher-forced top-1 agreement with the authors' forward on 20 pilot-table
    prompts (>= 95% of steps with margin > 0.1) and prompt-pass max KL < 0.1."""
    import tiktoken

    spec = importlib.util.spec_from_file_location(
        "v", ROOT / "vendor" / "chronogpt" / "ChronoGPT_instruct.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = mod.ChronoGPT(**config)
    ref.load_state_dict({k: v.float() for k, v in sd.items()})
    ref = ref.to("cuda").eval()
    enc = tiktoken.get_encoding("gpt2")
    table = tb.build_arm_a("pilot", ["preflight"], 2, rhos=(0.5,))
    toks = [
        enc.encode(chronogpt_wrap(t, p))
        for t, p in zip(table.input_text, table.prefill, strict=True)
    ][:20]
    agree = n = 0
    kl_max = 0.0
    for s in toks:
        a = mod.generate(ref, torch.tensor([s], device="cuda"), 8, 1792, temperature=0.0)
        a = a[0, len(s) :].tolist()
        lengths = torch.tensor([len(s)])
        lg0, cache = fast.prefill(cf.Prompts(torch.tensor([s]), lengths))
        gen = fast.new_gen_cache(1, 1, 8)
        got = [lg0[0]]
        for st in range(len(a) - 1):
            t = torch.tensor([[a[st]]], device="cuda")
            got.append(
                fast.decode(t, (lengths[:, None] + st).cuda(), cache, lengths, gen, st)[0, 0]
            )
        for st in range(len(got)):
            want = ref(torch.tensor([s + a[:st]], device="cuda"))[0, -1]
            if st == 0:
                p, q = torch.log_softmax(want, -1), torch.log_softmax(got[0], -1)
                kl_max = max(kl_max, float((p.exp() * (p - q)).sum()))
            top2 = torch.topk(want, 2).values
            if float(top2[0] - top2[1]) > 0.1:
                n += 1
                agree += int(want.argmax() == got[st].argmax())
    del ref
    torch.cuda.empty_cache()
    share = agree / n if n else 1.0
    return dict(
        top1_agree=share,
        steps=n,
        prefill_kl_max=kl_max,
        passed=bool(share >= 0.95 and kl_max < 0.1),
    )


def check_table(name: str, table) -> None:
    """The table about to be generated must be the one frozen in configs/generation.yaml."""
    want = CFG["tables"][name]["sha256"]
    have = tb.table_sha256(table)
    if have != want:
        raise RuntimeError(f"{name}: prompt table {have[:12]} != frozen {want[:12]}")


def run_chrono(model: str, table=None, run_name: str | None = None) -> None:
    run_name = run_name or model
    if table is None:
        table = session.model_table(model)
        check_table(model, table)
    if (GEN / run_name).exists() and len(Ledger(GEN / "ledger.jsonl").completed(run_name)) == (
        session.n_shards(table)
    ):
        log(f"{run_name}: complete, skipping")
        return
    t0 = time.time()
    sd, config, rev, path = load_chrono(model)
    fast = cf.FastChrono(sd, config, device="cuda", dtype=torch.bfloat16)
    log(f"{model}: loaded in {time.time() - t0:.0f}s")
    pf_path = GEN / "preflight" / f"{model}.json"
    if not pf_path.exists():
        pf = preflight(sd, config, fast)
        pf_path.parent.mkdir(parents=True, exist_ok=True)
        pf_path.write_text(json.dumps(pf, indent=2))
    pf = json.loads(pf_path.read_text())
    if not pf["passed"]:
        Ledger(GEN / "ledger.jsonl").append(event="preflight_failed", model=model, **pf)
        log(f"{model}: PREFLIGHT FAILED {pf}; not generated")
        return
    be = ChronoBackend(fast, rev, run_name)
    t1 = time.time()
    run_shards(be, table, GEN, gen_config(), Ledger(GEN / "ledger.jsonl"), batch_size=BATCH)
    log(f"{run_name}: {len(table)} prompts in {time.time() - t1:.0f}s")
    cap = GEN / "capability" / f"{model}.json"
    if run_name == model and not cap.exists():
        cap.parent.mkdir(parents=True, exist_ok=True)
        cap.write_text(json.dumps(hs.score(hs.items(), hs.chrono_loglik(fast)), indent=2))
    del fast, sd
    torch.cuda.empty_cache()
    shutil.rmtree(path.parents[2], ignore_errors=True)  # the model's HF cache directory


def run_ladder(model: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = CFG["models"][model]
    table = session.model_table(model)
    check_table(model, table)
    if len(Ledger(GEN / "ledger.jsonl").completed(model)) == session.n_shards(table):
        log(f"{model}: complete, skipping")
        return
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    lm = (
        AutoModelForCausalLM.from_pretrained(
            m["repo"], revision=m["revision"], dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    be = HFBackend(lm, tok, m["revision"], model, chat=True)
    run_shards(be, table, GEN, gen_config(), Ledger(GEN / "ledger.jsonl"), batch_size=8)
    cap = GEN / "capability" / f"{model}.json"
    if not cap.exists():
        cap.parent.mkdir(parents=True, exist_ok=True)
        cap.write_text(json.dumps(hs.score(hs.items(), hs.hf_loglik(lm, tok)), indent=2))
    del lm
    torch.cuda.empty_cache()


def compute_gate() -> dict:
    g = R.gate_family(GEN, session.CONFIRM)
    g["substitutions"] = R.substitutions_from(g)
    (GEN / "gate_session.json").write_text(json.dumps(g, indent=2, default=str))
    return g


def main() -> int:
    binding.check_binding(ROOT)  # the pod runs the frozen code or nothing
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "session_env.json").write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(0),
                cuda=torch.version.cuda,
                torch=torch.__version__,
                python=platform.python_version(),
                head=binding.head(ROOT),
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
            indent=2,
        )
    )
    for m in PRIORITY:
        run_chrono(m)
    g = compute_gate()
    log(f"gate: priority passing {g['priority_passing']}/6")
    if g["stop_rule_triggered"]:
        (GEN / "STOP").write_text("fewer than 3 of 6 priority vintages passed the gate\n")
        log("STOP RULE TRIGGERED: report to AJ")
        return 3
    for m in OTHERS:
        run_chrono(m)
    g = compute_gate()
    for failed, sub in g["substitutions"].items():
        run_chrono(sub, session.substitution_rows(failed), run_name=f"{sub}-for-{failed}")
    for m in session.LADDER:
        try:
            run_ladder(m)
        except Exception as e:  # descriptive only; a gated or failing ladder model is skipped
            log(f"{m}: ladder model skipped ({type(e).__name__}: {e})")
    binding.build_generation_manifest(GEN)
    (GEN / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
    log("SESSION DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
