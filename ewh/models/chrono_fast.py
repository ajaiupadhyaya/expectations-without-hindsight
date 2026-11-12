"""Batched, KV-cached sampler for ChronoGPT (plan change 7, Task 3.3).

Re-implements `ChronoGPT.forward` from vendor/chronogpt/ChronoGPT_instruct.py operation by
operation (same dtype casts, same 0-dim lambda multiplications, same SDPA call for the prompt),
and adds what the authors' `generate` lacks: right-padded batches, a per-layer key/value cache,
last-position logits only, and K continuations per prompt that attend to the shared prompt cache
by broadcasting (no K-fold copy). Points where a re-implementation can silently diverge, each
covered by tests/test_chrono_fast.py:
  - rotary positions restart at 0 in the authors' forward; decoding needs absolute offsets;
  - `is_causal=True` is wrong for one query against a cache (explicit masks are used instead);
  - value embeddings are indexed by token id and mixed into v before caching (layers 0-2, L-3..L-1);
  - half the rotary frequencies are zero; logits are capped at 15*tanh(x/15) in the model dtype.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

EOS = 50256
MASK64 = (1 << 64) - 1


def rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------- counter-based random numbers
def prompt_base(master_seed: int, model_revision: str, prompt_id: str) -> int:
    """64-bit stream key for one (seed, model, prompt): no two prompts share uniforms (plan D3)."""
    h = hashlib.blake2b(f"{master_seed}|{model_revision}|{prompt_id}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def _splitmix64(z: np.ndarray) -> np.ndarray:
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(MASK64)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(MASK64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(MASK64)
    return z ^ (z >> np.uint64(31))


def uniforms(bases: np.ndarray, draws: np.ndarray, step: int) -> np.ndarray:
    """(B, R) uniforms in [0, 1) for prompt keys `bases` (B,), draw indices `draws` (R,), step."""
    with np.errstate(over="ignore"):
        b = bases.astype(np.uint64)[:, None]
        d = (draws.astype(np.uint64) + np.uint64(1)) * np.uint64(0xD1B54A32D192ED03)
        s = np.uint64(step + 1) * np.uint64(0xAEF17502108EF2D9)
        z = _splitmix64(_splitmix64(b ^ d[None, :]) ^ s)
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-53


# ---------------------------------------------------------------- model
@dataclass
class Prompts:
    tokens: torch.Tensor  # (B, T) right-padded
    lengths: torch.Tensor  # (B,)


class FastChrono:
    def __init__(self, state_dict: dict, config: dict, device="cpu", dtype=torch.bfloat16):
        self.D, self.H, self.L = config["model_dim"], config["num_heads"], config["num_layers"]
        self.V = config["vocab_size"]
        self.Dh = self.D // self.H
        self.half = self.L // 2
        self.dtype, self.device = dtype, torch.device(device)

        def w(k):  # matrices and embeddings in the activation dtype (CastedLinear casts to it)
            return state_dict[k].to(self.device, dtype)

        def p(k):  # scalar mixing weights keep the parameter dtype, as in the authors' module
            return state_dict[k].to(self.device, torch.float32)

        self.embed = w("embed.weight")
        self.ve = [w(f"value_embeds.embed.{i}.weight") for i in range(3)]
        self.blocks = []
        for i in range(self.L):
            b = f"blocks.{i}."
            self.blocks.append(
                dict(
                    q=w(b + "attn.c_q.weight"),
                    k=w(b + "attn.c_k.weight"),
                    v=w(b + "attn.c_v.weight"),
                    o=w(b + "attn.c_proj.weight"),
                    attn_l=p(b + "attn.lambdas"),
                    fc=w(b + "mlp.c_fc.weight"),
                    proj=w(b + "mlp.c_proj.weight"),
                    blk_l=p(b + "lambdas"),
                )
            )
        self.skip = p("skip_weights")
        self.lm_head = w("lm_head.weight")
        L = self.L
        self.ve_layer = [i if i < 3 else (i - (L - 3) if i >= L - 3 else None) for i in range(L)]
        f = (1 / 1024) ** torch.linspace(0, 1, steps=self.Dh // 4, dtype=torch.float32)
        self.freq = torch.cat([f, f.new_zeros(self.Dh // 4)]).to(self.device)

    # ---------------------------------------------------------- pieces shared by both passes
    def _rotary(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x (..., H, Dh) with integer positions pos (...): the authors' Rotary at absolute pos."""
        theta = pos.to(torch.float32)[..., None] * self.freq  # (..., Dh/2)
        cos, sin = theta.cos()[..., None, :], theta.sin()[..., None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x)

    def _qkv(self, i: int, x: torch.Tensor, ve: torch.Tensor | None, pos: torch.Tensor):
        b = self.blocks[i]
        shape = (*x.shape[:-1], self.H, self.Dh)
        q = F.linear(x, b["q"]).view(shape)
        k = F.linear(x, b["k"]).view(shape)
        v = F.linear(x, b["v"]).view(shape)
        lam = b["attn_l"]
        v = lam[0] * v + lam[1] * ve.view_as(v) if ve is not None else lam[0] * v
        q, k = rms(q), rms(k)
        return self._rotary(q, pos), self._rotary(k, pos), v

    def _embed(self, tokens):
        return self.embed[tokens]

    def _ve(self, tokens):
        return [self.ve[j][tokens] for j in range(3)]

    def _logits(self, x_last):
        logits = F.linear(rms(x_last), self.lm_head)
        return (15 * torch.tanh(logits / 15)).float()

    # ---------------------------------------------------------- prompt pass
    @torch.inference_mode()
    def prefill(self, pr: Prompts):
        """Full causal pass over right-padded prompts. Returns last-position logits (B, V) and
        the per-layer cache [(k, v)] with k, v (B, T, H, Dh). Padding sits after each prompt's
        last token, so causal attention keeps it out of every real position."""
        x, cache = self._forward_all(pr.tokens.to(self.device), keep_cache=True)
        B = x.shape[0]
        x_last = x[torch.arange(B, device=self.device), pr.lengths.to(self.device) - 1]
        return self._logits(x_last), cache

    @torch.inference_mode()
    def continuation_logprob(self, seqs: list[list[int]], starts: list[int]) -> np.ndarray:
        """Sum of log-probabilities of tokens seqs[b][starts[b]:] given the tokens before them
        (HellaSwag scoring). Right-padded batch; only the needed positions get logits."""
        B = len(seqs)
        T = max(len(s) for s in seqs)
        tok = torch.zeros((B, T), dtype=torch.long)
        for b, s in enumerate(seqs):
            tok[b, : len(s)] = torch.tensor(s)
        x, _ = self._forward_all(tok.to(self.device), keep_cache=False)
        out = np.zeros(B)
        for b, (s, st) in enumerate(zip(seqs, starts, strict=True)):
            lp = torch.log_softmax(self._logits(x[b, st - 1 : len(s) - 1]), -1)
            target = torch.tensor(s[st:], device=self.device)
            out[b] = float(lp.gather(1, target[:, None]).sum())
        return out

    def _forward_all(self, tok: torch.Tensor, keep_cache: bool):
        """Final hidden states (B, T, D) of a full causal pass, and the per-layer cache."""
        B, T = tok.shape
        pos = torch.arange(T, device=self.device).expand(B, T)
        x0 = rms(self._embed(tok))
        x = x0
        ves = self._ve(tok)
        cache, skips = [], []
        for i in range(self.L):
            if i >= self.half:
                x = x + self.skip[i - self.half] * skips.pop()
            ve = ves[self.ve_layer[i]] if self.ve_layer[i] is not None else None
            q, k, v = self._qkv(i, rms(self._premix(i, x, x0)), ve, pos)
            if keep_cache:
                cache.append((k, v))
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
            )
            y = y.transpose(1, 2).contiguous().view(B, T, self.D)
            x = self._after_attn(i, x, x0, y)
            if i < self.half:
                skips.append(x)
        return x, cache

    def _premix(self, i, x, x0):
        lam = self.blocks[i]["blk_l"]
        return lam[0] * x + lam[1] * x0

    def _after_attn(self, i, x, x0, y):
        b = self.blocks[i]
        x = self._premix(i, x, x0) + F.linear(y, b["o"])
        h = F.relu(F.linear(rms(x), b["fc"])).square()
        return x + F.linear(h, b["proj"])

    # ---------------------------------------------------------- decode step
    @torch.inference_mode()
    def decode(self, tok, pos, prompt_cache, lengths, gen_cache, g):
        """One step for (B, R) rows. tok, pos (B, R). Rows of prompt b attend to its prompt cache
        (positions < lengths[b]) and to their own generated keys 0..g. gen_cache[i] = (kg, vg)
        with shape (B, R, G, H, Dh) is filled at index g. Returns logits (B, R, V)."""
        B, R = tok.shape
        x0 = rms(self._embed(tok))
        x = x0
        ves = self._ve(tok)
        Tp = prompt_cache[0][0].shape[1]
        pad = torch.arange(Tp, device=self.device)[None, :] >= lengths.to(self.device)[:, None]
        scale = 1.0 / np.sqrt(self.Dh)
        skips = []
        for i in range(self.L):
            if i >= self.half:
                x = x + self.skip[i - self.half] * skips.pop()
            ve = ves[self.ve_layer[i]] if self.ve_layer[i] is not None else None
            q, k, v = self._qkv(i, rms(self._premix(i, x, x0)), ve, pos)
            kg, vg = gen_cache[i]
            kg[:, :, g] = k
            vg[:, :, g] = v
            kp, vp = prompt_cache[i]
            qf = q.float()
            sp = torch.einsum("brhd,bthd->brht", qf, kp.float()) * scale
            sp = sp.masked_fill(pad[:, None, None, :], float("-inf"))
            sg = torch.einsum("brhd,brghd->brhg", qf, kg[:, :, : g + 1].float()) * scale
            pr = torch.softmax(torch.cat([sp, sg], -1), -1)
            y = torch.einsum("brht,bthd->brhd", pr[..., :Tp], vp.float()) + torch.einsum(
                "brhg,brghd->brhd", pr[..., Tp:], vg[:, :, : g + 1].float()
            )
            y = y.to(self.dtype).reshape(B, R, self.D)
            x = self._after_attn(i, x, x0, y)
            if i < self.half:
                skips.append(x)
        return self._logits(x)

    def new_gen_cache(self, B, R, G):
        shape = (B, R, G, self.H, self.Dh)
        return [
            (
                torch.zeros(shape, dtype=self.dtype, device=self.device),
                torch.zeros(shape, dtype=self.dtype, device=self.device),
            )
            for _ in range(self.L)
        ]


# ---------------------------------------------------------------- generation
def stop_tables(decode_token, vocab: int):
    """Per-token flags for the stop rule (plan D3: strip leading whitespace, cut at the first
    newline): has_nonspace, has_newline, newline_after_text (within the token itself)."""
    nonspace = np.zeros(vocab, bool)
    newline = np.zeros(vocab, bool)
    nl_after = np.zeros(vocab, bool)
    for t in range(vocab):
        try:
            s = decode_token(t)
        except Exception:
            continue
        nonspace[t] = bool(s.strip())
        newline[t] = "\n" in s
        stripped = s.lstrip()
        nl_after[t] = bool(stripped) and "\n" in stripped
    return nonspace, newline, nl_after


@dataclass
class Generation:
    tokens: np.ndarray  # (B, R, max_new) generated ids
    first_top_ids: np.ndarray  # (B, 64)
    first_top_logprobs: np.ndarray  # (B, 64)


@torch.inference_mode()
def generate(
    model: FastChrono,
    prompts: list[list[int]],
    bases: np.ndarray,
    n_draws: int,
    temperature: float = 1.0,
    max_new_tokens: int = 12,
    greedy: bool = True,
    tables=None,
    pad_id: int = 0,
) -> Generation:
    """Row 0 of each prompt is the greedy read (if greedy); rows 1..n_draws are samples drawn by
    inverse CDF with counter-based uniforms keyed on (prompt base, draw index, step)."""
    B = len(prompts)
    lengths = torch.tensor([len(p) for p in prompts])
    T = int(lengths.max())
    tok = torch.full((B, T), pad_id, dtype=torch.long)
    for b, p in enumerate(prompts):
        tok[b, : len(p)] = torch.tensor(p)
    logits0, cache = model.prefill(Prompts(tok, lengths))
    lp0 = torch.log_softmax(logits0, -1)
    top = torch.topk(lp0, k=min(64, lp0.shape[-1]), dim=-1)

    R = n_draws + (1 if greedy else 0)
    draws = np.arange(n_draws)
    dev = model.device
    tabs = None if tables is None else [torch.as_tensor(t, device=dev) for t in tables]
    out = torch.zeros((B, R, max_new_tokens), dtype=torch.long)
    started = torch.zeros((B, R), dtype=torch.bool, device=dev)
    done = torch.zeros((B, R), dtype=torch.bool, device=dev)
    gen_cache = model.new_gen_cache(B, R, max_new_tokens)

    def choose(logits, step):
        nxt = torch.empty((B, R), dtype=torch.long, device=dev)
        if greedy:
            nxt[:, 0] = logits[:, 0].argmax(-1)
        probs = torch.softmax(logits[:, int(greedy) :].float() / temperature, -1)
        cdf = probs.cumsum(-1)
        u = torch.as_tensor(uniforms(bases, draws, step), dtype=torch.float32, device=dev)
        idx = torch.searchsorted(cdf, (u * cdf[..., -1]).unsqueeze(-1)).squeeze(-1)
        nxt[:, int(greedy) :] = idx.clamp_max(logits.shape[-1] - 1)
        return nxt

    nxt = choose(logits0[:, None, :].expand(B, R, -1), 0)
    for s in range(max_new_tokens):
        out[:, :, s] = nxt.cpu()
        if tabs is not None:
            nonspace, newline, nl_after = tabs
            ended = (started & newline[nxt]) | nl_after[nxt] | (nxt == EOS)
            started |= nonspace[nxt]
            done |= ended
        else:
            done |= nxt == EOS
        if s == max_new_tokens - 1 or bool(done.all()):
            break
        pos = (lengths.to(dev)[:, None] + s).expand(B, R)
        logits = model.decode(nxt, pos, cache, lengths, gen_cache, s)
        nxt = choose(logits, s + 1)
    return Generation(out.numpy(), top.indices.cpu().numpy(), top.values.cpu().numpy())


def load_state_dict(path, dtype=torch.bfloat16) -> dict:
    """Memory-mapped load of a pytorch_model.bin, converted tensor by tensor (a 7.4 GB fp32
    checkpoint fits a 16 GB machine as 3.7 GB of bf16)."""
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    return {
        k: (v.to(dtype) if v.is_floating_point() and v.ndim >= 2 else v.clone())
        for k, v in sd.items()
    }
