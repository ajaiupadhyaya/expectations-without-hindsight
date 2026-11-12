"""Zero-shot HellaSwag on a seeded 1,000-item validation subset: one capability score for every
model, ChronoGPT included (plan D15). Scoring follows lm-evaluation-harness: the query is
"{activity}: {ctx_a} {Ctx_b}" and each ending is scored by the summed log-probability of " "+ending;
acc picks the highest sum, acc_norm the highest sum per byte of the ending."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ewh.provenance.archive import RAW_ROOT
from ewh.sim.ar1 import stream

PATH = RAW_ROOT / "hellaswag" / "validation-00000-of-00001.parquet"
N_ITEMS = 1000


def _clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def items(n: int = N_ITEMS) -> pd.DataFrame:
    """The pre-registered subset: n items drawn without replacement with the `subsets` seed."""
    d = pd.read_parquet(PATH)
    idx = np.sort(stream("subsets", "hellaswag").choice(len(d), size=n, replace=False))
    d = d.iloc[idx].reset_index(drop=True)
    query = [
        _clean(f"{a}: {ca} {cb.capitalize()}")
        for a, ca, cb in zip(d.activity_label, d.ctx_a, d.ctx_b, strict=True)
    ]
    return pd.DataFrame(
        {
            "ind": d["ind"].astype(int),
            "query": query,
            "endings": [[_clean(e) for e in es] for es in d.endings],
            "label": d.label.astype(int),
        }
    )


def score(items_df: pd.DataFrame, loglik) -> dict:
    """loglik(contexts: list[str], continuations: list[str]) -> np.ndarray of summed logprobs."""
    ctxs, conts, item_of = [], [], []
    for i, row in items_df.iterrows():
        for e in row.endings:
            ctxs.append(row.query)
            conts.append(" " + e)
            item_of.append(i)
    ll = np.asarray(loglik(ctxs, conts), float)
    n_end = np.array([len(c.encode()) for c in conts], float)
    acc = acc_norm = 0
    k = 0
    for _, row in items_df.iterrows():
        m = len(row.endings)
        s = ll[k : k + m]
        acc += int(np.argmax(s) == row.label)
        acc_norm += int(np.argmax(s / n_end[k : k + m]) == row.label)
        k += m
    n = len(items_df)
    return dict(acc=acc / n, acc_norm=acc_norm / n, n=n)


def chrono_loglik(model, batch: int = 32):
    """Scorer for ewh.models.chrono_fast.FastChrono (tiktoken gpt2)."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    def f(ctxs, conts):
        out = []
        for i in range(0, len(ctxs), batch):
            seqs, starts = [], []
            for c, e in zip(ctxs[i : i + batch], conts[i : i + batch], strict=True):
                a, b = enc.encode(c), enc.encode(e)
                seqs.append(a + b)
                starts.append(len(a))
            out.extend(model.continuation_logprob(seqs, starts))
        return np.array(out)

    return f


def hf_loglik(model, tokenizer, batch: int = 16):
    """Scorer for transformers causal LMs (plain text, no chat template, as lm-eval does)."""
    import torch

    def f(ctxs, conts):
        out = []
        for c, e in zip(ctxs, conts, strict=True):
            a = tokenizer(c, add_special_tokens=False)["input_ids"]
            b = tokenizer(e, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([a + b], device=model.device)
            with torch.no_grad():
                lp = torch.log_softmax(model(ids).logits[0, len(a) - 1 : -1].float(), -1)
            out.append(float(lp.gather(1, torch.tensor(b, device=model.device)[:, None]).sum()))
        return np.array(out)

    return f
