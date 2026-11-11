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

_PHIL = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data"


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
        # Phase 0.4 access checks. Philadelphia Fed links carry a `hash=` that changes when a
        # file is updated; Phase 2 scrapes the landing page for the current link. Snapshots
        # are stored under their retrieval date so later quarters never overwrite them.
        Source(
            "spf_individual_rgdp_20261111",
            f"{_PHIL}/survey-of-professional-forecasters/data-files/files/Individual_RGDP.xlsx"
            "?sc_lang=en&hash=5D704094EC8A5F6A1E12D87808753982",
            "spf/2026-11-11/Individual_RGDP.xlsx",
        ),
        Source(
            "spf_release_dates_20261111",
            f"{_PHIL}/survey-of-professional-forecasters/spf-release-dates.txt"
            "?sc_lang=en&hash=CE16E2057464DBD7A139FFE6188B48EC",
            "spf/2026-11-11/spf-release-dates.txt",
        ),
        Source(
            "spf_documentation_20261111",
            f"{_PHIL}/survey-of-professional-forecasters/spf-documentation.pdf"
            "?sc_lang=en&hash=8408A4F1BF351A3C268B40F6BC7B95AA",
            "spf/2026-11-11/spf-documentation.pdf",
        ),
        Source(
            "rtdsm_routput_qvqd_20261111",
            f"{_PHIL}/real-time-data/data-files/xlsx/ROUTPUTQvQd.xlsx"
            "?sc_lang=en&hash=34FA1C6BF0007996E1885C8C32E3BEF9",
            "rtdsm/2026-11-11/ROUTPUTQvQd.xlsx",
        ),
        # FRED: 3-month T-bill (TB3MS), never revised; CSV download needs no API key.
        Source(
            "fred_tb3ms_20261111",
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS",
            "fred/2026-11-11/TB3MS.csv",
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
