"""GPU pilot (plan Task 3.8): full-weight sampler equivalence and throughput on the NON-instruct
base model manelalab/chrono-gpt-v1-19991231 (outside every confirmatory set), pilot seeds, rho 0.5.

Computes only what the plan allows on a ChronoGPT-family model before the freeze: sampler
equivalence against the authors' code, parse rate, tokens/prompts per second, load time. No
forecast or gate statistic. Writes pilot_gpu.json in the working directory.
Run on the pod:  uv run python pod/pilot_gpu.py
"""

from __future__ import annotations

import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from scipy import stats

from ewh.models import chrono_fast as cf
from ewh.models.backend import ChronoBackend
from ewh.prompts import tables as tb
from ewh.prompts.parse import parse_number
from ewh.prompts.templates import chronogpt_wrap
from ewh.sim import ar1

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text())
M = CFG["models"]["chrono-base-1999"]
SEED = ar1.seeds()["pilot"]


def vendored():
    spec = importlib.util.spec_from_file_location(
        "ChronoGPT_instruct", ROOT / "vendor" / "chronogpt" / "ChronoGPT_instruct.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    out = dict(
        model=M["repo"],
        revision=M["revision"],
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,
        python=platform.python_version(),
    )
    t0 = time.time()
    bin_path = hf_hub_download(M["repo"], "pytorch_model.bin", revision=M["revision"])
    cfg_path = hf_hub_download(M["repo"], "config.pt", revision=M["revision"])
    out["download_seconds"] = time.time() - t0
    raw_cfg = torch.load(cfg_path, map_location="cpu", weights_only=True)
    out["config_pt"] = {k: str(v) for k, v in raw_cfg.items()}
    # config.pt also carries keys the constructor does not take (e.g. model_type)
    config = {k: raw_cfg[k] for k in ("vocab_size", "num_layers", "num_heads", "model_dim")}
    t1 = time.time()
    sd = torch.load(bin_path, map_location="cpu", mmap=True, weights_only=True)
    fast = cf.FastChrono(sd, config, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    out["load_seconds"] = time.time() - t1

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    table = tb.build_arm_a("pilot", ["chrono-base-1999"], 20, rhos=(0.5,))
    texts = [chronogpt_wrap(t, p) for t, p in zip(table.input_text, table.prefill, strict=True)]
    toks = [enc.encode(s, allowed_special={"<|endoftext|>"}) for s in texts]

    # 1. Prompt-pass logits and greedy tokens against the authors' class (fp32 params, bf16 compute)
    mod = vendored()
    ref = mod.ChronoGPT(**config)
    ref.load_state_dict({k: v.float() for k, v in sd.items()})
    ref = ref.to("cuda").eval()
    kls, agree, n_cmp = [], 0, 0
    for s in toks[:20]:
        x = torch.tensor([s], device="cuda")
        want = ref(x)[0, -1]
        got, _ = fast.prefill(cf.Prompts(torch.tensor([s]), torch.tensor([len(s)])))
        p, q = torch.log_softmax(want, -1), torch.log_softmax(got[0], -1)
        kls.append(float((p.exp() * (p - q)).sum()))
        top2 = torch.topk(want, 2).values
        if float(top2[0] - top2[1]) > 0.1:
            n_cmp += 1
            agree += int(want.argmax() == got[0].argmax())
    out["prefill_kl_max"] = max(kls)
    out["prefill_top1_agree"] = f"{agree}/{n_cmp}"

    # 2. Greedy continuations: authors' generate vs ours (12 tokens, 50 prompts)
    tables = cf.stop_tables(lambda t: enc.decode([t]), enc.n_vocab)
    same = 0
    for s in toks[:50]:
        a = mod.generate(ref, torch.tensor([s], device="cuda"), 12, 1792, temperature=0.0)
        a = a[0, len(s) :].tolist()
        g = cf.generate(
            fast,
            [s],
            np.array([1], np.uint64),
            n_draws=0,
            max_new_tokens=12,
            greedy=True,
            tables=tables,
        )
        b = list(g.tokens[0, 0, : len(a)])
        same += int(a[: len(b)] == b[: len(a)])
    out["greedy_identical"] = f"{same}/50"

    # 3. Sampled numbers: authors' sampler vs ours, two-sample KS on parsed first answers
    torch.manual_seed(SEED % 2**31)
    theirs, ours = [], []
    for s, row in zip(toks[:200], table.itertuples(), strict=False):
        a = mod.generate(ref, torch.tensor([s], device="cuda"), 12, 1792, temperature=1.0)
        v = parse_number(enc.decode(a[0, len(s) :].tolist()), row.fmt, str(row.prefill).rstrip(":"))
        theirs.append(v)
    del ref
    torch.cuda.empty_cache()
    be = ChronoBackend(fast, M["revision"], "chrono-base-1999")
    for i in range(0, 200, 25):
        chunk = table.iloc[i : i + 25]
        sb = be.sample(
            list(chunk.prompt_id),
            list(zip(chunk.input_text, chunk.prefill, strict=True)),
            1,
            SEED,
            1.0,
            12,
        )
        for txt, row in zip(sb.texts, chunk.itertuples(), strict=True):
            ours.append(parse_number(txt[1], row.fmt, str(row.prefill).rstrip(":")))
    th = np.array([v for v in theirs if v is not None])
    ou = np.array([v for v in ours if v is not None])
    out["parse_rate_theirs"] = len(th) / len(theirs)
    out["parse_rate_ours"] = len(ou) / len(ours)
    out["ks_p_sampled_numbers"] = (
        float(stats.ks_2samp(th, ou).pvalue) if len(th) and len(ou) else None
    )
    Path("pilot_gpu_partial.json").write_text(json.dumps(out, indent=2, default=str))

    # 4. Throughput at the planned K, by batch size
    thr = {}
    for k in (16, 32):
        for bs in (16, 32, 64):
            batch = list(zip(table.input_text[:bs], table.prefill[:bs], strict=True))
            ids = list(table.prompt_id[:bs])
            be.sample(ids[:4], batch[:4], k, SEED, 1.0, 12)  # warm-up
            torch.cuda.synchronize()
            t = time.time()
            reps = 3
            for _ in range(reps):
                be.sample(ids, batch, k, SEED, 1.0, 12)
            torch.cuda.synchronize()
            thr[f"K{k}_B{bs}"] = bs * reps / (time.time() - t)
    out["prompts_per_second"] = thr
    out["max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    Path("pilot_gpu.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
