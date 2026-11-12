"""Generation backends behind one protocol (plan "Core interfaces").

Every backend returns, for each prompt, K + 1 texts: index 0 is the greedy read and 1..K are
samples at the configured temperature (no top-k, no top-p, no repetition penalty). Texts are the
raw continuations after the prefill; parsing happens later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ewh.prompts.templates import chronogpt_wrap


@dataclass
class SampleBatch:
    texts: list[list[str]]  # (B, K + 1); [b][0] is the greedy read
    n_prompt_tokens: list[int]
    first_top_ids: np.ndarray | None = None  # (B, 64)
    first_top_logprobs: np.ndarray | None = None


class Backend(Protocol):
    name: str
    revision: str

    def sample(
        self,
        prompt_ids: list[str],
        prompts: list[tuple[str, str]],
        k: int,
        master_seed: int,
        temperature: float,
        max_new_tokens: int,
    ) -> SampleBatch: ...


class ChronoBackend:
    """ChronoGPT-Instruct through the batched cached sampler."""

    def __init__(self, model, revision: str, name: str):
        import tiktoken

        from ewh.models import chrono_fast as cf

        self.cf, self.model, self.revision, self.name = cf, model, revision, name
        self.enc = tiktoken.get_encoding("gpt2")
        self.tables = cf.stop_tables(lambda t: self.enc.decode([t]), self.enc.n_vocab)

    def _text(self, ids: np.ndarray) -> str:
        ids = list(ids)
        if self.cf.EOS in ids:
            ids = ids[: ids.index(self.cf.EOS)]
        return self.enc.decode([i for i in ids if i < self.enc.n_vocab])

    def sample(self, prompt_ids, prompts, k, master_seed, temperature, max_new_tokens):
        toks = [
            self.enc.encode(chronogpt_wrap(text, prefill), allowed_special={"<|endoftext|>"})
            for text, prefill in prompts
        ]
        bases = np.array(
            [self.cf.prompt_base(master_seed, self.revision, p) for p in prompt_ids],
            dtype=np.uint64,
        )
        g = self.cf.generate(
            self.model,
            toks,
            bases,
            n_draws=k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            greedy=True,
            tables=self.tables,
        )
        texts = [[self._text(g.tokens[b, r]) for r in range(k + 1)] for b in range(len(toks))]
        return SampleBatch(texts, [len(t) for t in toks], g.first_top_ids, g.first_top_logprobs)


class HFBackend:
    """Transformers causal LMs: the CPU stand-in (gpt2, Alpaca-style wrapper) and the chat
    models of the capability ladder (their own chat template, no system prompt, prefill after
    the assistant header). Sampling arguments are passed explicitly so a model's
    generation_config.json (e.g. Qwen's top_k, top_p, repetition_penalty) is never used."""

    SAMPLING = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0)

    def __init__(self, model, tokenizer, revision: str, name: str, chat: bool):
        self.model, self.tok, self.revision, self.name, self.chat = (
            model,
            tokenizer,
            revision,
            name,
            chat,
        )

    def render(self, text: str, prefill: str) -> str:
        if not self.chat:
            return chronogpt_wrap(text, prefill)
        head = self.tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
        return head + prefill

    def generation_kwargs(self, temperature: float, max_new_tokens: int, greedy: bool) -> dict:
        if greedy:
            return dict(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.0,
                temperature=None,
                top_k=None,
                top_p=None,
            )
        return {**self.SAMPLING, "temperature": temperature, "max_new_tokens": max_new_tokens}

    def sample(self, prompt_ids, prompts, k, master_seed, temperature, max_new_tokens):
        import torch

        from ewh.models.chrono_fast import prompt_base

        texts, ntok = [], []
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        for pid, (text, prefill) in zip(prompt_ids, prompts, strict=True):
            enc = self.tok(self.render(text, prefill), return_tensors="pt").to(self.model.device)
            n = enc["input_ids"].shape[1]
            ntok.append(n)
            row = []
            with torch.no_grad():
                g = self.model.generate(
                    **enc,
                    pad_token_id=pad,
                    **self.generation_kwargs(temperature, max_new_tokens, True),
                )
                row.append(self.tok.decode(g[0, n:], skip_special_tokens=True))
                # per-prompt stream: reproducible regardless of batching, unrelated across prompts
                torch.manual_seed(prompt_base(master_seed, self.revision, pid) % (2**63))
                s = self.model.generate(
                    **enc,
                    pad_token_id=pad,
                    num_return_sequences=k,
                    **self.generation_kwargs(temperature, max_new_tokens, False),
                )
            row += [self.tok.decode(s[j, n:], skip_special_tokens=True) for j in range(k)]
            texts.append(row)
        return SampleBatch(texts, ntok)


class PlantedBackend:
    """Synthetic forecaster with known error-on-revision coefficient and noise, for the dry run
    (plan Task 3.6). It looks up each prompt's unit in the prompt table rather than reading the
    text, and emits numbers as text so the parser is exercised too.

    Systematic forecasts (Arm A, known rho): F*_{t-1} = rho^2 x_{t-1}; F*_t = F*_{t-1} +
    lam (rho x_t - F*_{t-1}), lam = 1/(1 + theta); draws add N(0, noise_sd^2)."""

    def __init__(self, table, theta: float, noise_sd: float, name="planted", revision="planted"):
        self.name, self.revision = name, revision
        self.theta, self.noise_sd = theta, noise_sd
        self.sys: dict[str, float] = {}
        lam = 1.0 / (1.0 + theta)
        for _, g in table.groupby("unit_id"):
            tm1, t = g[g.date == "tm1"].iloc[0], g[g.date == "t"].iloc[0]
            rho, x_tm1, x_t = float(t.rho), float(tm1.x_last), float(t.x_last)
            f_tm1 = rho**2 * x_tm1
            self.sys[tm1.prompt_id] = f_tm1
            self.sys[t.prompt_id] = f_tm1 + lam * (rho * x_t - f_tm1)

    def sample(self, prompt_ids, prompts, k, master_seed, temperature, max_new_tokens):
        from scipy.stats import norm

        from ewh.models.chrono_fast import prompt_base, uniforms

        texts = []
        for pid in prompt_ids:
            f = self.sys[pid]
            u = uniforms(
                np.array([prompt_base(master_seed, self.revision, pid)], np.uint64), np.arange(k), 0
            )[0]
            draws = f + self.noise_sd * norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
            texts.append([f" {f:.0f}"] + [f" {d:.3f}" for d in draws])
        return SampleBatch(texts, [0] * len(prompt_ids))


class DryRunBackend:
    """Synthetic forecaster for the dry run of the whole session and the confirm scripts: planted
    Arm A units (known coefficient theta) and the least-squares learner elsewhere (gate and Arm B
    rows), all with noise. Never used on confirmatory seeds."""

    def __init__(self, table, name: str, theta: float = -0.3, noise_sd: float = 10.0):
        self.name, self.revision, self.noise_sd = name, f"dry-{name}", noise_sd
        armA = table[table.block.isin(["main", "crossed"])]
        self.planted = PlantedBackend(armA, theta, noise_sd) if len(armA) else None
        from ewh.sim.benchmarks import ls_learner

        hs = table["h"].fillna(1).astype(int) if "h" in table else [1] * len(table)
        self.fallback = {
            pid: float(ls_learner(np.array([[float(v) for v in h.split(",")]]), int(hh))[0])
            for pid, h, hh in zip(table.prompt_id, table["hist"], hs, strict=True)
        }

    def sample(self, prompt_ids, prompts, k, master_seed, temperature, max_new_tokens):
        from scipy.stats import norm

        from ewh.models.chrono_fast import prompt_base, uniforms

        texts = []
        for pid in prompt_ids:
            if self.planted is not None and pid in self.planted.sys:
                f, sd = self.planted.sys[pid], self.noise_sd
            else:
                f, sd = self.fallback[pid], 0.3
            u = uniforms(
                np.array([prompt_base(master_seed, self.revision, pid)], np.uint64), np.arange(k), 0
            )[0]
            draws = f + sd * norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
            texts.append([f" {f:.2f}"] + [f" {d:.3f}" for d in draws])
        return SampleBatch(texts, [0] * len(prompt_ids))
