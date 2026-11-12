import numpy as np
import pandas as pd
import pytest

from ewh.models.backend import HFBackend, PlantedBackend
from ewh.models.ledger import Ledger
from ewh.models.runner import DRAW_COLUMNS, GenerationConfig, read_draws, run_shards
from ewh.prompts import tables as tb


@pytest.fixture
def table():
    return tb.build_arm_a("dryrun", ["m"], 3, rhos=(0.6,))  # 24 templates x 3 paths x 2 dates


CFG = GenerationConfig(k=4, temperature=1.0, max_new_tokens=12, master_seed=7, shard_size=20)


def test_resume_after_crash_equals_uninterrupted(tmp_path, table):
    be = PlantedBackend(table, theta=-0.3, noise_sd=5.0)
    run_shards(be, table, tmp_path / "a", CFG, Ledger(tmp_path / "a.jsonl"))
    full = read_draws(tmp_path / "a", "planted")

    led = Ledger(tmp_path / "b.jsonl")
    run_shards(be, table.iloc[:20], tmp_path / "b", CFG, led)  # shard 0 completes
    with pytest.raises(RuntimeError):  # the run dies halfway through shard 1
        run_shards(be, table, tmp_path / "b", CFG, led, batch_size=5, stop_after_rows=10)
    run_shards(be, table, tmp_path / "b", CFG, led)
    resumed = read_draws(tmp_path / "b", "planted")
    pd.testing.assert_frame_equal(
        full.drop(columns="code_commit"), resumed.drop(columns="code_commit")
    )
    events = [(e["event"], e["shard"]) for e in led.entries() if "shard" in e]
    assert ("failed", 1) in events and ("redo", 1) in events
    assert led.verify() == []
    assert not (tmp_path / "b" / "planted" / "shard_00001.tmp").exists()


def test_completed_shards_are_never_regenerated(tmp_path, table):
    be = PlantedBackend(table, theta=0.0, noise_sd=1.0)
    led = Ledger(tmp_path / "l.jsonl")
    assert run_shards(be, table, tmp_path, CFG, led) == -(-len(table) // CFG.shard_size)
    assert run_shards(be, table, tmp_path, CFG, led) == 0


def test_schema_and_draw_indices(tmp_path, table):
    be = PlantedBackend(table, theta=0.0, noise_sd=1.0)
    run_shards(be, table, tmp_path, CFG, Ledger(tmp_path / "l.jsonl"))
    d = read_draws(tmp_path, "planted")
    assert list(d.columns) == DRAW_COLUMNS
    assert len(d) == len(table) * (CFG.k + 1)
    assert set(d.draw) == {-1, 0, 1, 2, 3}
    assert d.parsed.notna().all()


def test_ledger_detects_tampering(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.append(event="start", model="m", shard=0)
    led.append(event="done", model="m", shard=0, sha256="x")
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    lines[0] = lines[0].replace('"start"', '"done"')
    (tmp_path / "l.jsonl").write_text("\n".join(lines) + "\n")
    assert led.verify() != []


def test_hf_backend_overrides_model_generation_config():
    """Qwen ships top_k=20, top_p=0.8, repetition_penalty=1.05 defaults; ours must not apply."""
    be = HFBackend(None, None, "r", "x", chat=True)
    kw = be.generation_kwargs(1.0, 12, greedy=False)
    assert kw["top_k"] == 0 and kw["top_p"] == 1.0 and kw["repetition_penalty"] == 1.0
    assert kw["temperature"] == 1.0 and kw["do_sample"] is True
    g = be.generation_kwargs(1.0, 12, greedy=True)
    assert g["do_sample"] is False and g["repetition_penalty"] == 1.0


def test_hf_backend_end_to_end_tiny_model(tmp_path, table):
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(list("0123456789-., :\nabcdefghijklmnopqrstuvwxyz#()"))}
    vocab["<unk>"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<unk>")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_layer=1, n_embd=16, n_head=2,
                                       n_positions=4096))  # fmt: skip
    be = HFBackend(model, tok, "rev", "tiny", chat=False)
    cfg = GenerationConfig(k=3, temperature=1.0, max_new_tokens=4, master_seed=1, shard_size=10)
    run_shards(be, table.iloc[:4], tmp_path, cfg, Ledger(tmp_path / "l.jsonl"))
    d1 = read_draws(tmp_path, "tiny")
    assert len(d1) == 4 * 4
    # same seed -> same samples (per-prompt streams)
    run_shards(be, table.iloc[:4], tmp_path / "again", cfg, Ledger(tmp_path / "l2.jsonl"))
    d2 = read_draws(tmp_path / "again", "tiny")
    assert list(d1.raw_text) == list(d2.raw_text)
    assert np.all(d1.groupby("prompt_id").raw_text.nunique() >= 1)
