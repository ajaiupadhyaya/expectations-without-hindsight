"""Tag binding, input verification and the single-attempt marker (plan Task 4.2, D17, change 12).

Adapted from sharp-markets `src/sharpmkt/confirm.py` (check_binding, create_marker, write_once),
with input hashing added: raw data through data/raw/MANIFEST.json, generations through
data/generations/MANIFEST.json and the hash-chained ledger, and the exact pre-registered shard set.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ewh.models.ledger import Ledger
from ewh.provenance import manifest
from ewh.provenance.archive import REPO_ROOT, sha256_file

TAG = "pap-v1.2"  # Deviations 1-2 (2026-11-12); pap-v1 and pap-v1.1 are kept
BOUND = ("ewh", "vendor", "prompts", "configs", "pod", "pyproject.toml", "uv.lock")
RESULTS = REPO_ROOT / "docs" / "results"
GEN_ROOT = REPO_ROOT / "data" / "generations"


class Refused(RuntimeError):
    """A binding or guard check failed before any outcome was read."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def check_binding(repo: Path = REPO_ROOT, tag: str = TAG, bound: tuple[str, ...] = BOUND):
    """Return T0 (the tag's tagger date, UTC) or raise Refused. Requires an annotated tag that
    HEAD descends from, and every bound file byte-identical to its blob at the tag, with nothing
    modified or untracked under the bound paths."""
    kind = _git(repo, "cat-file", "-t", tag, check=False)
    if kind.returncode != 0:
        raise Refused(f"no tag {tag!r}")
    if kind.stdout.strip() != "tag":
        raise Refused(f"{tag!r} is a lightweight tag; the plan needs an annotated tag")
    if _git(repo, "merge-base", "--is-ancestor", tag, "HEAD", check=False).returncode != 0:
        raise Refused(f"HEAD does not descend from {tag}")
    tagged = _git(repo, "ls-tree", "-r", "--name-only", tag, "--", *bound).stdout.split()
    for path in tagged:
        want = _git(repo, "rev-parse", f"{tag}:{path}").stdout.strip()
        f = repo / path
        have = _git(repo, "hash-object", str(f)).stdout.strip() if f.exists() else "missing"
        if have != want:
            raise Refused(f"{path} differs from its blob at {tag} ({have[:12]} != {want[:12]})")
    if _git(repo, "diff", "--quiet", tag, "--", *bound, check=False).returncode != 0:
        raise Refused(f"tracked changes since {tag} under {', '.join(bound)}")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *bound).stdout
    if dirty.strip():
        raise Refused(f"modified or untracked files under {', '.join(bound)}:\n{dirty}")
    date = _git(repo, "for-each-ref", f"refs/tags/{tag}", "--format=%(taggerdate:iso-strict)")
    return datetime.fromisoformat(date.stdout.strip()).astimezone(UTC)


# ---------------------------------------------------------------- generations manifest
def build_generation_manifest(root: Path = GEN_ROOT) -> dict:
    """Index every shard parquet (path, sha256, bytes) by model. Written once after the session
    and committed before any estimate (plan Task 6.0)."""
    models = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(d.glob("*.parquet"))
        models[d.name] = {
            f.name: {"sha256": sha256_file(f), "bytes": f.stat().st_size} for f in files
        }
    out = {"models": models}
    (root / "MANIFEST.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def check_inputs(
    expected_shards: dict[str, int],
    gen_root: Path = GEN_ROOT,
    raw_root: Path | None = None,
) -> None:
    """Refuse unless (i) the raw archive verifies, (ii) the ledger chain is intact, (iii) each model
    has exactly shards 0..n-1 completed (no missing, extra or replaced shard), and (iv) every
    shard file matches both the ledger and the generations manifest."""
    if raw_root is not None:
        problems = manifest.verify(raw_root)
        if problems:
            raise Refused("raw archive: " + "; ".join(problems[:5]))
    ledger = Ledger(gen_root / "ledger.jsonl")
    if ledger.verify():
        raise Refused("generation ledger chain broken: " + "; ".join(ledger.verify()[:3]))
    mpath = gen_root / "MANIFEST.json"
    if not mpath.exists():
        raise Refused("missing data/generations/MANIFEST.json")
    man = json.loads(mpath.read_text())["models"]
    for model, n in expected_shards.items():
        done_events = [
            e for e in ledger.entries() if e.get("model") == model and e.get("event") == "done"
        ]
        by_shard: dict[int, set[str]] = {}
        for e in done_events:
            by_shard.setdefault(e["shard"], set()).add(e["sha256"])
        if set(by_shard) != set(range(n)):
            missing = sorted(set(range(n)) - set(by_shard))
            extra = sorted(set(by_shard) - set(range(n)))
            raise Refused(f"{model}: shards missing {missing[:5]} extra {extra[:5]}")
        replaced = [s for s, h in by_shard.items() if len(h) > 1]
        if replaced:
            raise Refused(f"{model}: shards completed twice with different contents {replaced}")
        for s, hashes in by_shard.items():
            name = f"shard_{s:05d}.parquet"
            f = gen_root / model / name
            if not f.exists():
                raise Refused(f"{model}/{name} missing")
            h = sha256_file(f)
            if h not in hashes or man.get(model, {}).get(name, {}).get("sha256") != h:
                raise Refused(f"{model}/{name} does not match the ledger and manifest")


# ---------------------------------------------------------------- one attempt per family
def create_marker(family: str, results: Path = RESULTS, head: str = "") -> Path:
    """Exclusive creation before any outcome is read; a second attempt refuses."""
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{family}.attempt"
    try:
        with open(path, "x") as f:
            f.write(f"attempt started {datetime.now(UTC).isoformat(timespec='seconds')} {head}\n")
    except FileExistsError as e:
        raise Refused(f"{path} exists: the {family} run has already been attempted") from e
    return path


def write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(text)


def head(repo: Path = REPO_ROOT) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()
