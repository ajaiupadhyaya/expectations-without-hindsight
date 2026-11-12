"""Pilot follow-up: greedy equivalence without early stopping, and teacher-forced per-step
agreement of the cached decode with the authors' uncached forward (full weights, base model)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download

from ewh.models import chrono_fast as cf
from ewh.prompts import tables as tb
from ewh.prompts.templates import chronogpt_wrap

ROOT = Path(__file__).resolve().parents[1]
M = yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text())["models"]["chrono-base-1999"]


def main():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    raw = torch.load(
        hf_hub_download(M["repo"], "config.pt", revision=M["revision"]),
        map_location="cpu",
        weights_only=True,
    )
    config = {k: raw[k] for k in ("vocab_size", "num_layers", "num_heads", "model_dim")}
    sd = torch.load(
        hf_hub_download(M["repo"], "pytorch_model.bin", revision=M["revision"]),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    fast = cf.FastChrono(sd, config, device="cuda", dtype=torch.bfloat16)
    spec = importlib.util.spec_from_file_location(
        "v", ROOT / "vendor/chronogpt/ChronoGPT_instruct.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = mod.ChronoGPT(**config)
    ref.load_state_dict({k: v.float() for k, v in sd.items()})
    ref = ref.to("cuda").eval()
    table = tb.build_arm_a("pilot", ["chrono-base-1999"], 20, rhos=(0.5,))
    toks = [
        enc.encode(chronogpt_wrap(t, p))
        for t, p in zip(table.input_text, table.prefill, strict=True)
    ][:30]
    same, first_div, kls, top1, n_top = 0, [], [], 0, 0
    for s in toks:
        a = mod.generate(ref, torch.tensor([s], device="cuda"), 12, 1792, temperature=0.0)[
            0, len(s) :
        ].tolist()
        g = cf.generate(
            fast,
            [s],
            np.array([1], np.uint64),
            n_draws=0,
            max_new_tokens=12,
            greedy=True,
            tables=None,
        )
        b = [int(x) for x in g.tokens[0, 0, :12]]
        same += int(a == b)
        div = next((i for i in range(12) if a[i] != b[i]), 12)
        first_div.append(div)
        # teacher-forced on the authors' continuation
        lengths = torch.tensor([len(s)])
        lg0, cache = fast.prefill(cf.Prompts(torch.tensor([s]), lengths))
        gen = fast.new_gen_cache(1, 1, 12)
        got = [lg0[0]]
        for step in range(11):
            t = torch.tensor([[a[step]]], device="cuda")
            got.append(
                fast.decode(t, (lengths[:, None] + step).cuda(), cache, lengths, gen, step)[0, 0]
            )
        for step in range(12):
            want = ref(torch.tensor([s + a[:step]], device="cuda"))[0, -1]
            p, q = torch.log_softmax(want, -1), torch.log_softmax(got[step], -1)
            kls.append(float((p.exp() * (p - q)).sum()))
            t2 = torch.topk(want, 2).values
            if float(t2[0] - t2[1]) > 0.1:
                n_top += 1
                top1 += int(want.argmax() == got[step].argmax())
    out = dict(
        greedy_identical_12_tokens=f"{same}/30",
        first_divergence_step=first_div,
        teacher_forced_kl_mean=float(np.mean(kls)),
        teacher_forced_kl_max=float(max(kls)),
        teacher_forced_top1_agree=f"{top1}/{n_top}",
    )
    Path("greedy_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
