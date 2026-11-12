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

## Deviation 2 (2026-11-12): stop-rule tables sized to the model's vocabulary; tag `pap-v1.2`
- **Results seen:** none. Session attempt 1 (pod `wknwaqrygxqk91`, from `pap-v1.1`, 04:02 UTC)
  loaded chrono-1999, passed the preflight (an allowed diagnostic: top-1 agreement 0.994 on 157
  steps, max prompt KL 0.009) and crashed in shard 0 before any draw was written. Its log, ledger
  and preflight file are kept in `data/session_attempt1/` (not used for estimation).
- **Problem (a bug in `pap-v1.1`):** ChronoGPT's embedding and output layer have 50,304 rows; the
  gpt2 tokenizer has 50,257 ids. The sampler draws from the full softmax, as the authors' code does,
  so a padding id (50,257–50,303) can be sampled. `ChronoBackend._text` already drops such ids, but
  the stop-rule tables had only 50,257 entries, so a sampled padding id indexed past their end
  (CUDA device-side assert). The CPU dry run used a synthetic backend and never hit this path.
- **Change:** `chrono_fast.generate` pads the stop tables to the model's vocabulary with False, so a
  padding id carries no text and no stop flag. The sampling distribution, seeds, stop rule, parser,
  estimators, tables and allocation are unchanged. `binding.TAG` becomes `pap-v1.2`.
- **Tests:** `test_ids_beyond_the_tokenizer_vocab_do_not_crash_the_stop_rule` (fails on
  `pap-v1.1`), and `test_chrono_backend_end_to_end_on_real_prompts` (slow): ChronoBackend through
  `run_shards` on real prompts from every block, then HellaSwag scoring, on a tiny random model
  with the full 50,304-row vocabulary. It also fails on `pap-v1.1` and passes now.
- **Session restart:** attempt 2 starts from an empty `data/generations/` on the pod.
