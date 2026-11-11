"""Philadelphia Fed (SPF, RTDSM) and FRED snapshot downloads.

Links on philadelphiafed.org carry a `hash=` that changes when a file is updated, so each file is
located by scraping its landing page for the current link. All files of one snapshot are stored
under data/raw/<source>/<SNAPSHOT>/ (Decision 2). `python -m ewh.data.philfed` fetches the
snapshot; files already archived are not downloaded again.
"""

from __future__ import annotations

import html
import re
import sys
import time

import httpx

from ewh.provenance import archive, manifest

SNAPSHOT = "2026-11-11"
SITE = "https://www.philadelphiafed.org"
RT = f"{SITE}/surveys-and-data/real-time-data-research"

# (landing page, filename regex, archive subdirectory)
FILES: list[tuple[str, str, str]] = [
    (f"{RT}/individual-forecasts", r"SPFmicrodata\.xlsx", "spf"),
    (f"{RT}/survey-of-professional-forecasters", r"spf-release-dates\.txt", "spf"),
    (f"{RT}/noutput", r"NOUTPUTQvQd\.xlsx", "rtdsm"),
    (f"{RT}/routput", r"ROUTPUTQvQd\.xlsx", "rtdsm"),
    (f"{RT}/p", r"PQvQd\.xlsx", "rtdsm"),
    (f"{RT}/rcon", r"RCONQvQd\.xlsx", "rtdsm"),
    (f"{RT}/rinvbf", r"(?i)rinvbfQvQd\.xlsx", "rtdsm"),
    (f"{RT}/rinvresid", r"(?i)rinvresidQvQd\.xlsx", "rtdsm"),
    (f"{RT}/rgf", r"(?i)rgfQvQd\.xlsx", "rtdsm"),
    (f"{RT}/rgsl", r"(?i)rgslQvQd\.xlsx", "rtdsm"),
    (f"{RT}/ruc", r"(?i)rucQvMd\.xlsx", "rtdsm"),
    (f"{RT}/cpi", r"(?i)cpiQvMd\.xlsx", "rtdsm"),
    (f"{RT}/ipt", r"(?i)iptMvMd\.xlsx", "rtdsm"),
    (f"{RT}/hstarts", r"(?i)hstartsMvMd\.xlsx", "rtdsm"),
    (f"{RT}/hstarts", r"(?i)hstarts_first_second_third\.xlsx", "rtdsm"),
]
FRED = ["TB3MS", "GS10"]


def find_link(http: httpx.Client, page: str, pattern: str) -> str:
    r = http.get(page)
    r.raise_for_status()
    hrefs = [html.unescape(h) for h in re.findall(r'href="([^"]+)"', r.text)]
    hits = sorted({h for h in hrefs if re.search(pattern + r"(\?|$)", h.split("/")[-1])})
    if len(hits) != 1:
        raise RuntimeError(f"{page}: expected one link matching {pattern}, found {hits}")
    return hits[0] if hits[0].startswith("http") else SITE + hits[0]


def fetch_snapshot() -> None:
    with archive.client() as http:
        for page, pattern, sub in FILES:
            url = find_link(http, page, pattern)
            name = url.split("?")[0].split("/")[-1]
            side = archive.fetch(url, archive.RAW_ROOT / sub / SNAPSHOT / name, client=http)
            print(f"{sub}/{name}: {side['bytes']} bytes")
            time.sleep(1.0)
        for s in FRED:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
            side = archive.fetch(
                url, archive.RAW_ROOT / "fred" / SNAPSHOT / f"{s}.csv", client=http
            )
            print(f"fred/{s}.csv: {side['bytes']} bytes")
    manifest.build()


if __name__ == "__main__":
    fetch_snapshot()
    sys.exit(0)
