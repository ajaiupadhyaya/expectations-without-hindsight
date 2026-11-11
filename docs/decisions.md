# Decisions

One entry per methodological choice: date, choice, alternatives, reason. Design decisions
D1 to D17 and planning changes 1 to 16 are recorded in `docs/PLAN.md`; entries here add to or
refine them. After tag `pap-v1`, changes go to `docs/deviations.md` instead.

## Decision 1: Repository layout and raw-data policy (2026-11-11)
- **Choice:** package `ewh/` at the repo root (handoff Section 7). Raw payloads are not
  committed; each has a committed `.sidecar.json` (URL, UTC time, HTTP status, bytes, sha256)
  and `data/raw/MANIFEST.json` indexes them.
- **Alternatives:** commit raw files (AKLMT replication zip alone is about 115 MB); git-lfs.
- **Reason:** keeps the repo small and public-safe; hashes make every rebuild checkable; the
  raw archive is deposited in Phase 7.

## Decision 2: Philadelphia Fed snapshots and Excel engine (2026-11-11)
- **Choice:** store each Philadelphia Fed download under its retrieval date
  (`data/raw/<source>/<YYYY-MM-DD>/`), and read workbooks with the `calamine` engine.
- **Alternatives:** overwrite in place (violates append-only); `openpyxl` (fails on
  `ROUTPUTQvQd.xlsx`, malformed document-properties date).
- **Reason:** files are updated every quarter under the same name; the confirmatory run must
  read one frozen snapshot whose hashes are bound at `pap-v1`.

## Decision 3: Rate variables from FRED, AAA excluded (2026-11-11)
- **Choice:** TB3MS and GS10 from FRED (public domain, never revised). Moody's AAA is dropped.
- **Alternatives:** keep AAA under FRED's copyright notice (needs Moody's permission).
- **Reason:** plan change 14; FRED marks AAA as third-party copyright; the public SPF files also
  omit Moody's history.
