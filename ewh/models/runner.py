"""Resumable shard runner: prompt table -> one parquet per shard, every event in the ledger.

A shard is a fixed slice of the prompt table (order fixed by the table). It is redone under its
pre-committed seed if it started but did not finish; a completed shard is never regenerated.
Raw generations are the archived data (handoff Section 6).
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ewh.models.ledger import Ledger
from ewh.prompts.parse import parse_number

DRAW_COLUMNS = [
    "prompt_id",
    "block",
    "model",
    "model_revision",
    "code_commit",
    "master_seed",
    "draw",
    "temperature",
    "raw_text",
    "parsed",
    "shard",
]
PROMPT_COLUMNS = [
    "prompt_id",
    "model",
    "model_revision",
    "n_prompt_tokens",
    "first_top_ids",
    "first_top_logprobs",
    "device",
    "torch_version",
    "created_utc",
    "shard",
]


@dataclass(frozen=True)
class GenerationConfig:
    k: int
    temperature: float
    max_new_tokens: int
    master_seed: int
    shard_size: int


def code_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_label(row) -> str | None:
    return None if row.get("fmt") != "indexed" else str(row["prefill"]).rstrip(":")


def run_shards(
    backend,
    table: pd.DataFrame,
    out_dir: Path,
    cfg: GenerationConfig,
    ledger: Ledger,
    batch_size: int = 32,
    stop_after_rows: int | None = None,
) -> int:
    """Generate every shard not yet completed. Returns the number of shards run now.
    `stop_after_rows` simulates a crash (tests): raise after that many rows of a shard."""
    import torch

    out = Path(out_dir) / backend.name
    out.mkdir(parents=True, exist_ok=True)
    done = ledger.completed(backend.name)
    n_shards = (len(table) + cfg.shard_size - 1) // cfg.shard_size
    commit = code_commit()
    ran = 0
    for s in range(n_shards):
        path = out / f"shard_{s:05d}.parquet"
        if s in done and path.exists() and _sha256(path) == done[s]:
            continue
        redo = any(e.get("shard") == s and e.get("model") == backend.name for e in ledger.entries())
        ledger.append(
            event="redo" if redo else "start",
            model=backend.name,
            shard=s,
            master_seed=cfg.master_seed,
            revision=backend.revision,
            commit=commit,
        )
        part = table.iloc[s * cfg.shard_size : (s + 1) * cfg.shard_size]
        draws, prompts_run = [], []
        try:
            for i in range(0, len(part), batch_size):
                if stop_after_rows is not None and i >= stop_after_rows:
                    raise RuntimeError("simulated crash")
                chunk = part.iloc[i : i + batch_size]
                batch = backend.sample(
                    list(chunk.prompt_id),
                    list(zip(chunk.input_text, chunk.prefill, strict=True)),
                    cfg.k,
                    cfg.master_seed,
                    cfg.temperature,
                    cfg.max_new_tokens,
                )
                now = datetime.now(UTC).isoformat()
                for j, (_, row) in enumerate(chunk.iterrows()):
                    for r, text in enumerate(batch.texts[j]):
                        draws.append(
                            dict(
                                prompt_id=row.prompt_id,
                                block=row.get("block"),
                                model=backend.name,
                                model_revision=backend.revision,
                                code_commit=commit,
                                master_seed=cfg.master_seed,
                                draw=r - 1,
                                temperature=cfg.temperature,
                                raw_text=text,
                                parsed=parse_number(
                                    text, row.get("fmt", "comma"), _target_label(row)
                                ),
                                shard=s,
                            )
                        )
                    prompts_run.append(
                        dict(
                            prompt_id=row.prompt_id,
                            model=backend.name,
                            model_revision=backend.revision,
                            n_prompt_tokens=batch.n_prompt_tokens[j],
                            first_top_ids=None
                            if batch.first_top_ids is None
                            else batch.first_top_ids[j].tolist(),
                            first_top_logprobs=None
                            if batch.first_top_logprobs is None
                            else batch.first_top_logprobs[j].tolist(),
                            device=_device(),
                            torch_version=torch.__version__,
                            created_utc=now,
                            shard=s,
                        )
                    )
        except Exception as e:
            ledger.append(event="failed", model=backend.name, shard=s, error=repr(e)[:300])
            raise
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(draws, columns=DRAW_COLUMNS).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        ptmp = out / f"prompts_{s:05d}.tmp"
        pd.DataFrame(prompts_run, columns=PROMPT_COLUMNS).to_parquet(ptmp, index=False)
        os.replace(ptmp, out / f"prompts_{s:05d}.parquet")
        ledger.append(
            event="done", model=backend.name, shard=s, rows=len(draws), sha256=_sha256(path)
        )
        ran += 1
    return ran


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} | driver {torch.version.cuda}"
    except Exception:
        pass
    return f"cpu | {platform.processor() or platform.machine()}"


def read_draws(out_dir: Path, model: str) -> pd.DataFrame:
    files = sorted((Path(out_dir) / model).glob("shard_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
