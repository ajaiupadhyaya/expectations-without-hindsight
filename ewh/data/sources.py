"""Registry of public raw sources. `python -m ewh.data.sources [name ...]` archives them.

Every entry is fetched once through ewh.provenance.archive (append-only, sha256 sidecar).
Terms of use for each source are recorded in docs/data_terms.md.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from ewh.provenance import archive, manifest


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    dest: str  # relative to data/raw
    md5: str | None = None  # publisher checksum, when one is published


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        # Afrouzi, Kwon, Landier, Ma, Thesmar (2023 QJE) replication package, Harvard Dataverse
        # doi:10.7910/DVN/3Q6BHC, file id 6950610, CC0 1.0.
        Source(
            "aklmt_replication",
            "https://dataverse.harvard.edu/api/access/datafile/6950610",
            "aklmt/Replication.zip",
            md5="b335bac26bbfc7be61ea6c35180cb371",
        ),
    ]
}


def fetch(name: str, root: Path = archive.RAW_ROOT) -> dict:
    src = SOURCES[name]
    dest = root / src.dest
    side = archive.fetch(src.url, dest)
    if src.md5:
        got = hashlib.md5(dest.read_bytes()).hexdigest()
        if got != src.md5:
            raise RuntimeError(f"{name}: md5 {got} != published {src.md5}")
    return side


def main(argv: list[str]) -> int:
    names = argv or list(SOURCES)
    for n in names:
        side = fetch(n)
        print(f"{n}: {side['bytes']} bytes sha256 {side['sha256'][:12]}")
    manifest.build()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
