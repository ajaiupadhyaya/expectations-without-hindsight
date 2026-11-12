# Deviations from the pre-analysis plan

Every change to bound code, inputs or specification after tag `pap-v1` is logged here with: date,
what changed, why, and a **results seen** field (none / which results). A deviation made after
seeing a confirmatory result demotes that hypothesis to exploratory; the pre-registered result is
still reported as primary.

## Deviation 1 (2026-11-12): runs not generated leave the expected shard set; tag `pap-v1.1`
- **Results seen:** none. No ChronoGPT-instruct vintage had been downloaded, loaded or queried, and
  no confirmatory generation existed. Found while checking the session plan after AJ chose to drop
  the Llama ladder models (PAP A15).
- **Problem (a bug in `pap-v1`):** the confirm runner's input check (`ewh/confirm/run.py`, P3)
  required shards for every ChronoGPT vintage and every ladder model. But the PAP lets the session
  leave runs out: the Llama models are dropped without an HF token (A15), a ladder model is
  descriptive and was already skipped on error by `pod/run_session.py`, and a vintage that fails
  the preflight "is not generated and is reported" (Decision 11). Any of these would have made
  every confirmatory family refuse to run. A preflight-failed vintage also never entered the
  gate, so the D9 substitution rule could not cover its quarters.
- **Change (approved by AJ, 2026-11-12):** (1) `pod/run_session.py` writes a ledger event
  `skipped` (model, reason) when a ladder model fails; (2) `ewh.confirm.run.excluded_runs` leaves
  out of the expected shard set, and of every family, any model with a `preflight_failed` or
  `skipped` ledger event whose shards are not all complete; (3) the gate records a
  preflight-failed vintage as failing (`{"pass": false, "preflight_failed": true}`), so it counts
  against the stop rule and is substituted like any gate failure; (4) `binding.TAG` becomes
  `pap-v1.1`. Tests: `tests/test_deviation1.py`. Estimators, thresholds, templates, seeds, tables
  and the allocation (`configs/generation.yaml`) are unchanged.
- **Tags:** `pap-v1` (20bdaae) is kept as signed; `pap-v1.1` is the tag the session and the
  confirm scripts bind to. P1 of the PAP now reads `pap-v1.1`.
