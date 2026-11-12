"""Sampler equivalence tests on a tiny ChronoGPT with every parameter randomised.

The authors' constructor zero-initialises lm_head and MLP output weights and sets the block
lambdas to [1, 0]; with those values every logit is 0 and any test would pass vacuously, so all
parameters are drawn at random here (plan Task 3.3)."""

import importlib.util

import numpy as np
import pytest
import torch
from scipy import stats

from ewh.models import chrono_fast as cf
from ewh.provenance.archive import REPO_ROOT

CFG = dict(vocab_size=97, num_layers=6, num_heads=2, model_dim=32)


def vendored():
    spec = importlib.util.spec_from_file_location(
        "ChronoGPT_instruct", REPO_ROOT / "vendor" / "chronogpt" / "ChronoGPT_instruct.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref():
    torch.manual_seed(0)
    mod = vendored()
    m = mod.ChronoGPT(**CFG)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(
                torch.randn_like(p) * (0.5 if p.ndim >= 2 else 0.3) + (0.5 if p.ndim < 2 else 0)
            )
    m.eval()
    return m


def prompts(rng, lengths):
    return [list(rng.integers(0, CFG["vocab_size"], n)) for n in lengths]


def full_logits_fast(model, seqs):
    """Uncached reference: one prefill per sequence, last-position logits."""
    out = []
    for s in seqs:
        lg, _ = model.prefill(cf.Prompts(torch.tensor([s]), torch.tensor([len(s)])))
        out.append(lg[0])
    return torch.stack(out)


def forced_decode(model, pr, conts):
    """Teacher-forced cached decoding: pr[b] then conts[b][r] (all the same length G).
    Returns logits (B, R, G+1, V): step 0 from the prefill, step s from decode."""
    B, R, G = len(pr), len(conts[0]), len(conts[0][0])
    lengths = torch.tensor([len(p) for p in pr])
    tok = torch.zeros((B, int(lengths.max())), dtype=torch.long)
    for b, p in enumerate(pr):
        tok[b, : len(p)] = torch.tensor(p)
    lg0, cache = model.prefill(cf.Prompts(tok, lengths))
    out = [lg0[:, None, :].expand(B, R, -1)]
    gen = model.new_gen_cache(B, R, G)
    for s in range(G):
        t = torch.tensor([[conts[b][r][s] for r in range(R)] for b in range(B)])
        pos = (lengths[:, None] + s).expand(B, R)
        out.append(model.decode(t, pos, cache, lengths, gen, s))
    return torch.stack(out, 2)


def test_fp32_cached_equals_fp32_uncached(ref):
    model = cf.FastChrono(ref.state_dict(), CFG, dtype=torch.float32)
    rng = np.random.default_rng(1)
    pr = prompts(rng, [5, 11, 8])  # unequal lengths in one right-padded batch
    conts = [[list(rng.integers(0, 97, 4)) for _ in range(3)] for _ in pr]  # 3 rows, 4 steps
    got = forced_decode(model, pr, conts)
    for b in range(3):
        for r in range(3):
            seqs = [pr[b] + conts[b][r][:s] for s in range(5)]
            want = full_logits_fast(model, seqs)
            assert torch.allclose(got[b, r], want, atol=1e-4), (b, r)


def test_k_draws_independent(ref):
    """Changing one row's continuation never changes another row's logits."""
    model = cf.FastChrono(ref.state_dict(), CFG, dtype=torch.float32)
    rng = np.random.default_rng(2)
    pr = prompts(rng, [7])
    c1 = [[list(rng.integers(0, 97, 3)) for _ in range(3)]]
    c2 = [[c1[0][0], list(rng.integers(0, 97, 3)), c1[0][2]]]
    a, b = forced_decode(model, pr, c1), forced_decode(model, pr, c2)
    assert torch.equal(a[0, 0], b[0, 0]) and torch.equal(a[0, 2], b[0, 2])
    assert not torch.allclose(a[0, 1, 2:], b[0, 1, 2:])


def test_batch_invariance(ref):
    model = cf.FastChrono(ref.state_dict(), CFG, dtype=torch.float32)
    rng = np.random.default_rng(3)
    pr = prompts(rng, [6, 13, 9])
    conts = [[list(rng.integers(0, 97, 3))] for _ in pr]
    together = forced_decode(model, pr, conts)
    for b in range(3):
        alone = forced_decode(model, [pr[b]], [conts[b]])
        assert torch.allclose(together[b], alone[0], atol=1e-5)


def _kl(a, b):
    p, q = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return float((p.exp() * (p - q)).sum())


def test_bf16_matches_vendored_forward(ref):
    """bf16: the prompt pass reproduces the authors' forward; cached decode steps are at least as
    close to an fp32 reference as the authors' own bf16 forward is (our attention runs in fp32),
    and agree on the top token wherever the top-two margin exceeds 0.1."""
    b16 = cf.FastChrono(ref.state_dict(), CFG, dtype=torch.bfloat16)
    f32 = cf.FastChrono(ref.state_dict(), CFG, dtype=torch.float32)
    rng = np.random.default_rng(4)
    ours, theirs, agree, n = [], [], 0, 0
    for _ in range(12):
        pr = prompts(rng, [int(rng.integers(6, 15))])
        conts = [[list(rng.integers(0, 97, 4))]]
        got = forced_decode(b16, pr, conts)
        for s in range(5):
            seq = pr[0] + conts[0][0][:s]
            want = ref(torch.tensor([seq]))[0, -1]
            if s == 0:
                assert torch.allclose(got[0, 0, 0], want, atol=1e-2)
                continue
            truth = full_logits_fast(f32, [seq])[0]
            ours.append(_kl(truth, got[0, 0, s]))
            theirs.append(_kl(truth, want))
            top2 = torch.topk(want, 2).values
            if float(top2[0] - top2[1]) > 0.1:
                n += 1
                agree += int(want.argmax() == got[0, 0, s].argmax())
    assert np.mean(ours) <= np.mean(theirs) * 1.5 + 1e-3, (np.mean(ours), np.mean(theirs))
    assert max(ours) <= max(theirs) * 1.5 + 1e-3
    assert agree == n


def test_rng_streams():
    bases = np.array([cf.prompt_base(1, "rev", f"p{i}") for i in range(2000)], dtype=np.uint64)
    u = cf.uniforms(bases, np.arange(8), step=0)
    assert np.array_equal(u, cf.uniforms(bases, np.arange(8), step=0))  # reproducible
    assert stats.kstest(u.ravel(), "uniform").pvalue > 0.001
    assert abs(np.corrcoef(u[:, 0], u[:, 1])[0, 1]) < 0.06  # draws within a prompt
    # the t and t-1 prompts of a pair have different ids: their streams are unrelated
    tm1 = np.array([cf.prompt_base(1, "rev", f"q{i}") for i in range(2000)], dtype=np.uint64)
    assert abs(np.corrcoef(u[:, 0], cf.uniforms(tm1, np.arange(8), 0)[:, 0])[0, 1]) < 0.06
    assert not np.array_equal(u, cf.uniforms(bases, np.arange(8), step=1))
    assert cf.prompt_base(1, "rev", "p0") != cf.prompt_base(2, "rev", "p0")
    assert cf.prompt_base(1, "rev", "p0") != cf.prompt_base(1, "rev2", "p0")


def test_sampling_matches_softmax(ref):
    model = cf.FastChrono(ref.state_dict(), CFG, dtype=torch.float32)
    rng = np.random.default_rng(5)
    pr = prompts(rng, [8])
    n = 4000
    bases = np.array([cf.prompt_base(9, "rev", "x")], dtype=np.uint64)
    g = cf.generate(model, pr, bases, n_draws=n, max_new_tokens=1, greedy=True)
    first = g.tokens[0, 1:, 0]
    lg, _ = model.prefill(cf.Prompts(torch.tensor(pr), torch.tensor([8])))
    p = torch.softmax(lg[0], -1).numpy().astype(float)
    obs = np.bincount(first, minlength=97)
    keep = p * n >= 5
    exp = p[keep] * n
    o = np.append(obs[keep], obs[~keep].sum())
    e = np.append(exp, n - exp.sum())
    assert stats.chisquare(o, e * o.sum() / e.sum()).pvalue > 0.001
    assert g.tokens[0, 0, 0] == int(np.argmax(p))  # row 0 is the greedy read


def test_stop_rule_tables():
    vocab = ["\n", " 23", "4", "\n\n", " x\n", "\n5", " "]
    non, nl, after = cf.stop_tables(lambda t: vocab[t], len(vocab))
    assert list(non) == [False, True, True, False, True, True, False]
    assert list(after) == [False, False, False, False, True, False, False]
    assert nl[0] and nl[3] and not nl[1]
