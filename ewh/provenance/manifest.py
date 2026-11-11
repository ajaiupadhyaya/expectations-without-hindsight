"""Index and verify the raw archive: MANIFEST.json is rebuilt from the per-file sidecars.

Adapted from gpu-econ-life `src/gel/release/manifest.py` (build, verify).
Usage: python -m ewh.provenance.manifest build|verify [root]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ewh.provenance.archive import RAW_ROOT, SIDECAR_SUFFIX, sha256_file, sidecar_path

MANIFEST_NAME = "MANIFEST.json"
FIELDS = ("source_url", "retrieved_at", "bytes", "sha256")


def _payloads(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and not p.name.endswith(SIDECAR_SUFFIX)
        and p.name != MANIFEST_NAME
        and p.name != ".DS_Store"
    )


def build(root: Path = RAW_ROOT) -> dict:
    """Write root/MANIFEST.json from sidecars. Payloads without sidecars are skipped
    (verify reports them)."""
    files = {}
    for p in _payloads(root):
        side = sidecar_path(p)
        if not side.exists():
            continue
        meta = json.loads(side.read_text())
        files[p.relative_to(root).as_posix()] = {k: meta[k] for k in FIELDS}
    manifest = {"files": dict(sorted(files.items()))}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(root: Path = RAW_ROOT) -> list[str]:
    """Return a list of problems; empty means every archived file matches its recorded hash."""
    problems: list[str] = []
    mpath = root / MANIFEST_NAME
    if not mpath.exists():
        return [f"missing {MANIFEST_NAME}"]
    files = json.loads(mpath.read_text())["files"]
    for rel, meta in files.items():
        p = root / rel
        if not p.exists():
            problems.append(f"missing payload: {rel}")
            continue
        if p.stat().st_size != meta["bytes"] or sha256_file(p) != meta["sha256"]:
            problems.append(f"hash mismatch: {rel}")
    for p in _payloads(root):
        rel = p.relative_to(root).as_posix()
        if not sidecar_path(p).exists():
            problems.append(f"no sidecar: {rel}")
        elif rel not in files:
            problems.append(f"not in manifest: {rel}")
    return problems


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "verify"
    root = Path(argv[1]) if len(argv) > 1 else RAW_ROOT
    if cmd == "build":
        m = build(root)
        print(f"manifest: {len(m['files'])} files")
        return 0
    if cmd == "verify":
        problems = verify(root)
        for p in problems:
            print(p)
        print("raw archive OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
