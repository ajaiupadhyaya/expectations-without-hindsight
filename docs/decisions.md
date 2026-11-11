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
