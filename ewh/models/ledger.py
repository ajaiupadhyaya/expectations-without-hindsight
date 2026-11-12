"""Append-only, hash-chained generation ledger (plan change 12, D17).

Every shard start, completion, failure and redo is one JSON line whose `prev` field is the
sha256 of the previous line, so an edited or deleted line breaks the chain. The ledger is
pushed before any estimate is computed (plan Task 6.0).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

GENESIS = "0" * 64


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _lines(self) -> list[str]:
        return self.path.read_text().splitlines() if self.path.exists() else []

    def entries(self) -> list[dict]:
        return [json.loads(line) for line in self._lines()]

    def append(self, **event) -> dict:
        lines = self._lines()
        prev = hashlib.sha256(lines[-1].encode()).hexdigest() if lines else GENESIS
        entry = {"time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), **event, "prev": prev}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def verify(self) -> list[str]:
        problems, prev = [], GENESIS
        for i, line in enumerate(self._lines()):
            if json.loads(line).get("prev") != prev:
                problems.append(f"line {i + 1}: chain broken")
            prev = hashlib.sha256(line.encode()).hexdigest()
        return problems

    def completed(self, model: str) -> dict[int, str]:
        """shard -> sha256 of its parquet, for shards whose last event is 'done'."""
        state: dict[int, tuple[str, str | None]] = {}
        for e in self.entries():
            if e.get("model") == model and "shard" in e:
                state[e["shard"]] = (e["event"], e.get("sha256"))
        return {s: h for s, (ev, h) in state.items() if ev == "done" and h}
