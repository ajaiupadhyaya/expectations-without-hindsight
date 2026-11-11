"""Re-run the novelty-check OpenAlex queries (docs/novelty_check.md). Prints new hits.

Usage: uv run python scripts/novelty_openalex.py [from_date]
Run before tagging pap-v1 to close the SSRN blind spot.
"""

import sys
import time

import httpx

QUERIES = [
    "ChronoGPT",
    "chronologically consistent language model",
    "point-in-time language model",
    "DatedGPT",
    "large language model forecast revision overreaction",
    "language model overreaction",
    "large language model expectations formation",
    "ChatGPT forecasts extrapolation",
    "LLM forecasts noise",
    "LLM forecasts repeated draws",
    "look-ahead bias language model forecasts",
]
SSRN = "S4210172589"
UA = "expectations-without-hindsight/0.1 (academic research)"


def main(since: str) -> None:
    seen = set()
    with httpx.Client(headers={"User-Agent": UA}, timeout=60) as http:
        for q in QUERIES:
            for extra in ("", f",primary_location.source.id:{SSRN}"):
                r = http.get(
                    "https://api.openalex.org/works",
                    params={
                        "search": q,
                        "filter": f"from_publication_date:{since}{extra}",
                        "per-page": 15,
                        "select": "title,publication_date,doi",
                    },
                )
                r.raise_for_status()
                for w in r.json()["results"]:
                    key = w.get("doi") or w["title"]
                    if key not in seen:
                        seen.add(key)
                        print(f"{w['publication_date']} | {w['title'][:120]} | {w.get('doi')}")
                time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2024-06-01")
