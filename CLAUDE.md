# Expectations Without Hindsight: project memory

## Goal
Measure expectation bias (over/underreaction) in language-model forecasts after removing
three artefacts: training-data leakage (vintage ChronoGPT models), sampling noise (K draws,
split-sample IV) and weak models (competence gate, capability ladder). Output: a
pre-registered working paper, a one-command replication repo, a web summary. Author: AJ (solo).

Read `docs/HANDOFF.md` (the spec) and `docs/PLAN.md` (the approved execution plan) before
any work. Handoff Sections 3 and 4 are binding. Where the two disagree the handoff wins,
except for the numbered items under "What planning changed" in the plan.

## Non-negotiable rules
1. Never fabricate data, coefficients, dates or citations. Unknown or unreachable values stay
   missing and go into `docs/open_questions.md`.
2. `data/raw/` is append-only. Every raw file is written through `ewh.provenance.archive`,
   which refuses to overwrite and writes a `.sidecar.json` (source URL, UTC retrieval time,
   HTTP status, bytes, sha256). `data/raw/MANIFEST.json` is rebuilt from the sidecars.
3. No number in the paper is typed by hand. Every number traces to a script in `ewh/` or
   `paper/make_inputs.py` and a file in `data/`.
4. No gate threshold, template, estimator or sample rule changes after outcomes are seen.
   After tag `pap-v1`, every change is a logged deviation in `docs/deviations.md`.
5. No ChronoGPT **instruct** vintage is loaded or queried before `pap-v1`. Pilots use only
   pilot seeds, rho = 0.5 and non-confirmatory models, and are logged in `docs/pilot_log.md`.
6. RunPod: $50 total (storage included) is the hard cap. Get AJ's explicit go, quoting the
   hourly price, before creating any pod or volume. Stop pods when idle; confirm they are stopped.
7. Log every methodological choice in `docs/decisions.md` and every assumption in
   `docs/assumptions.md`. Ask AJ before adding a data source or changing a logged spec.
8. Polite HTTP: respect robots.txt and terms, at least 1 s between requests per host,
   backoff on 429/5xx. The User-Agent never carries personal details.
9. Stop points in the plan's phase table are real stops: report to AJ and wait.
10. Small commits after tests and ruff pass. Author `ajaiupadhyaya@users.noreply.github.com`.

## Stack
Python 3.12 via uv (never pip). numpy, pandas, pyarrow, scipy; httpx for downloads; pytest
and ruff. torch/tiktoken/transformers arrive in Phase 3. Paper: LaTeX via latexmk.

## Commands
- `make test`, `make lint`, `make all`
- `uv run python -m ewh.provenance.manifest build|verify`

## Conventions
- Sign convention: b < 0 overreaction, b > 0 underreaction.
- Dates UTC, ISO 8601. Quarters as `YYYYQn`.
- Seeds live in `configs/seeds.yaml` in disjoint namespaces (pilot, dryrun, power, gate,
  confirm, subsets).
