# Pre-analysis plan: Expectations Without Hindsight

**Status: SIGNED by AJ on 2026-11-12 ("Accept all": decisions A1–A17 as recommended). Frozen at
the annotated tag `pap-v1`.** From the tag on, every bound file is fixed and every change is a
logged deviation (`docs/deviations.md`).

Author: Ajai Upadhyaya. Drafted with Claude (Anthropic), 2026-11-12.
Spec: `docs/HANDOFF.md`. Execution plan: `docs/PLAN.md`. Decision log: `docs/decisions.md`.

---

## Decisions for AJ

AJ accepted every recommendation on 2026-11-12. The alternatives are kept for the record.

| # | Decision | Adopted (AJ, 2026-11-12) | Alternative (not adopted) |
|---|---|---|---|
| A1 | What "bias" means in Arm A (H2) | Contrast with the least-squares (LS) learner on the same paths (H2-LS) is confirmatory; H2 against zero is reported beside it. Reason: the LS learner, which fits an AR(1) to the 40 values shown, has b ≈ −0.28 at rho = 0.2 and ≈ +0.10 at rho = 1 (Decision 4), so b against zero mislabels learning as bias. | H2 against zero confirmatory |
| A2 | H1 direction | Two-sided. Noise pulls the one-draw coefficient toward about −1/2 from either side (`docs/noise_derivation.md`), so it can make a strong overreactor look milder. | One-sided (naive more negative) |
| A3 | Pooled Arm A estimand | Equal weights over rho ∈ {0.4, 0.6, 0.8, 1}; rho = 0 and 0.2 reported per cell with Fieller sets; rho² weights as robustness. Reason: the rho = 0.2 cell's instrument goes weak for near-rational forecasters (`docs/power.md`). | Equal weights over {0.2, …, 1}; rho² weights primary |
| A4 | Template inference (Arm A) | t-test across the 12 paraphrases of per-paraphrase estimates (11 df): the only candidate that holds its size when the coefficient varies by wording. | Cluster by path (over-rejects under heterogeneity) |
| A5 | H3 benchmark | H3 against zero (the LS learner's implied persistence is unbiased; Decision 4); H3-LS as robustness. | H3-LS confirmatory |
| A6 | Holm families | Arm A {H1, H2-LS, H3}; Arm B {H4, H5}. | Add H2 against zero to the Arm A family |
| A7 | Gate thresholds | Parse ≥ 95%; history-sensitivity slope ≥ 0.2 with one-sided p < 0.01; persistence slope ≥ 0.25 with 95% lower bound > 0; scale slope in [0.8, 1.2]. Validated: passes the LS learner and an AKLMT-style overreactor, fails echo, noisy echo and constant. | Other numbers, fixed now |
| A8 | Arm B framing | Named (variable and dates) primary; anonymous a pre-registered robustness row and part of the leakage secondary. | Anonymous primary |
| A9 | COVID | Drop pairs whose target quarter lies in 2020Q1–2021Q2 (primary); full sample as robustness. | Full sample primary |
| A10 | Arm B pooled inference | Block-t over 8 consecutive time blocks (7 df): holds size; Driscoll–Kraay over-rejects (13%) on 97 overlapping quarters. | Driscoll–Kraay with fixed-b |
| A11 | Variables | The 14 BGMS SPF series with real-time data; AAA dropped (Moody's copyright). | — |
| A12 | Stale or missing data | Pairs whose survey vintage ends early (2025Q4 for NIPA series; 2013Q4 and 2025Q4 for housing) or whose outcome was never published (UNEMP, t = 2025Q1) get no prompts (OQ2, Decision 13). | Keep stale pairs with a shorter history |
| A13 | Allocation | K = 32 draws + 1 greedy read; Arm A on all 26 vintages, 40 paths per template × rho; crossed block 20 paths per rho (1999, 2024); gate 100 per kind; Arm B all pre-registered runs; ladder 6 paths per template × rho. 959,916 prompts in 505 shards (`configs/generation.yaml`). | Fewer vintages with more paths |
| A14 | GPU and spend | Pick the GPU by live stock and price on the day (RTX 4090 ≈ $0.89/hr ≈ 11 h; H100 ≈ $3.99/hr ≈ 4 h): expected $10–$16 including storage, ceiling $35 for this session; ask before exceeding. | — |
| A15 | Llama ladder models | Gated: you add your Hugging Face token as a RunPod secret yourself, or the two Llama models are dropped (descriptive only). I never handle the token. | Drop them now |
| A16 | Visibility | Make the repository public when `pap-v1` is pushed (decided 2026-11-11; I will ask once more at that moment). | Keep private, register on OSF |
| A17 | Title | *Expectations Without Hindsight: Noise, Capability and Bias in Language-Model Forecasts* | — |

---

## 0. Conventions, disclosures and protocols

### 0.1 Conventions
- Sign: b < 0 overreaction, b > 0 underreaction. Error = outcome − forecast; revision = forecast
  at t − forecast at t−1 of the same target.
- All tests two-sided at 5% unless stated; every confirmatory result is reported with estimate,
  95% interval, two-sided p-value and minimum detectable effect (MDE), whether or not it rejects.
- "Vintage" Y means `manelalab/chrono-gpt-instruct-v1-Y1231` (training cutoff 31 December Y) at
  the revision pinned in `configs/generation.yaml`.

### 0.2 Disclosures: everything seen before the freeze
1. **Literature and data:** the novelty check (`docs/novelty_check.md`), verified references.
2. **Human benchmarks:** AKLMT (2023) Experiment 1 reproduced from their data (implied
   persistence 0.45–0.99; error-on-revision −0.48 at rho = 0 to −0.25 at rho = 1). BGMS (2020)
   Table 3 reproduced from raw SPF and RTDSM files for all 14 series; CG inflation coefficient
   1.41. **The first replication run failed** the pre-set CG band (1.52); three data-handling
   fixes followed from row-level diagnosis with the tolerance unchanged (Decision 8).
3. **Simulations:** planted forecasters, the LS learner's coefficients (Decision 4), size and power
   grids, an AR(4) benchmark panel for Arm B power whose pooled coefficient is −0.063
   (descriptive benchmark, not a hypothesis).
   **One exposure:** the 2026-11-12 dry run of the Arm B confirm script computed, on the real
   2000Q2–2025 data for RGDP and UNEMP only, the SPF consensus coefficient (−0.08), the SPF
   individual coefficient (−0.26) and the AR(4) benchmark coefficient (−0.33), pooled with the
   Arm B estimator. These are human and benchmark comparisons, not confirmatory hypotheses (H4
   and H5 concern model forecasts); dry runs no longer compute them.
4. **Model outputs before the freeze:** (i) pilot 1, Qwen2-0.5B-Instruct (not in any confirmatory
   set), templates v1, rho = 0.5, pilot seeds: parse rate, noise share, a weakly identified IV;
   (ii) pilot 2, the **non-instruct** ChronoGPT base model (1999), templates v1, pilot seeds:
   sampler equivalence, parse rate, throughput and load time only; no forecast or gate statistic;
   (iii) gpt2 on a dry-run gate table (fails the gate). No instruct vintage has been loaded.
5. **Templates:** v1 (6 paraphrases) were committed before any ChronoGPT output; v2 (12
   paraphrases) were added at AJ's request after pilot 2 with no outcome in view (Decision 12).
6. **Bugs found before the freeze** and fixed: delayed-release vintages (Decision 8), missing
   outcomes (Decision 13), seed-tag hashing that made Arm A paths identical across vintages
   (Decision 14). All were found by dry runs or replications, not by model outputs.
7. **AI use:** the code, simulations and this draft were written by Claude (Anthropic) under AJ's
   direction; AJ owns every decision listed above and signs the plan.
8. **Financial interests:** none.

### 0.3 Protocols
- **P1 Binding.** The session and every confirm script refuse to run unless HEAD descends from the
  annotated tag `pap-v1` and every file under `ewh/`, `vendor/`, `prompts/`, `configs/`, `pod/`,
  `pyproject.toml` and `uv.lock` is byte-identical to its blob at the tag with nothing modified or
  untracked (`ewh/confirm/binding.py`).
- **P2 One session.** One GPU session generates all raw draws (`pod/run_session.py`): six priority
  vintages, then the gate and its stop rule, then the other 20 vintages, then substitution runs,
  then the ladder. Every table's sha256 is checked against `configs/generation.yaml` before
  generation. Every shard start, completion, failure and redo is in a hash-chained ledger. A
  failed shard is redone under its pre-committed seed, never replaced with a new seed. Allowed
  in-session diagnostics: row counts, parse rate, tokens per second, the preflight, and the gate.
- **P3 One attempt per family.** `python -m ewh.confirm.run gate`, then `arm_a`, then `arm_b`. Each
  verifies binding, the raw archive, the ledger and the exact pre-registered shard set (nothing
  missing, extra or replaced), writes `docs/results/<family>.attempt`, and only then reads the
  generations. Results are written once; a failure writes its traceback and is reported.
- **P4 Before estimation.** `data/generations/MANIFEST.json` and the ledger are committed and
  pushed before the first confirm script runs.
- **P5 Deviations.** Any change after the tag is logged in `docs/deviations.md` with what changed,
  why, and which results had been seen. A deviation made after seeing a confirmatory result
  demotes that hypothesis to exploratory; the pre-registered result is still reported as primary.
- **P6 Stop rules.** Fewer than 3 of the 6 priority vintages pass the gate: the session writes
  STOP, the pod is stopped and AJ decides; the paper is reframed around the gate result rather
  than loosening it. Spend above the A14 ceiling: ask AJ first.

---

## 1. Data and measurement

### 1.1 Arm A: controlled forecasting experiment
- **Process (AKLMT 2023):** x_{s+1} = rho x_s + e, mean 0, sd(e) = 20, integers, rho ∈ {0, 0.2,
  0.4, 0.6, 0.8, 1}. Stationary start (random walk starts at 0).
- **Unit:** one path × one template × one model. Date t−1 shows 40 values and asks for the value
  two steps ahead; date t shows 41 values and asks for the next value (the same target).
- **Main block (nested):** for every vintage, template and rho, 40 fresh paths (confirm seeds).
  **Crossed block:** 20 paths per rho shown under all 24 templates, vintages 1999 and 2024.
- **Templates:** 12 paraphrases × 2 number formats (`prompts/templates.yaml`, v2, sha256 recorded
  in `configs/generation.yaml`). ChronoGPT prompts use the authors' instruction wrapper exactly;
  chat models use their own chat template with no system prompt.
- **Sampling:** temperature 1, no top-k or top-p, 12 new tokens, K = 32 draws plus one greedy read;
  random numbers keyed on (seed, model revision, prompt, draw, step) so no two prompts share them.
- **Parsing and validity (D6):** the first number in the first line of the answer (format-aware);
  a draw is invalid if no number parses or it lies more than 10 standard deviations of the shown
  history from the last shown value. A unit needs at least 8 valid draws at each date. The
  estimand is therefore the coefficient on means of truncated draws.

### 1.2 Arm B: real-time macro forecasts
- **Variables (14):** NGDP, RGDP, PGDP, RCONSUM, RNRESIN, RRESINV, RGF, RGSL, HOUSING, CPI,
  INDPROD, UNEMP, TB3M, TN10Y (`docs/decisions.md`, Decision 7).
- **Target:** four-quarter growth ending t+3 (levels for UNEMP and the two rates). **Outcome:**
  from the first vintage that contains x_{t+3} (dated t+4), numerator and denominator in one
  vintage. **Scaling:** each variable divided by its pre-2000 outcome s.d. (vintage 2000Q1).
- **Prompts:** the last 40 transformed values from the vintage of the survey quarter only, one
  decimal; t−1 asks 5 steps ahead, t asks 4. Named and anonymous framings; the same 24 templates.
- **Sample:** t = 2000Q2 to the last quarter with a published outcome (2025Q3 or 2025Q4 by series),
  minus decision A12's exclusions; 136,896 prompts per full run.
- **Pair rules:** primary, both forecasts of a pair from the vintage valid at t−1; robustness,
  latest valid vintage for each. A vintage that fails the gate is replaced in the rolling run by
  the latest earlier passing vintage (robustness: drop its quarters).
- **Runs:** annual rolling (primary), six-vintage rolling (fallback and robustness), and the 1999,
  2009 and 2024 vintages on every quarter.

### 1.3 Human and benchmark forecasts
- SPF consensus and individual forecasts built exactly as BGMS (`ewh/data/spf.py`, mode "armb");
  the LS learner and known-parameter rational forecaster (Arm A); a real-time AR(4) with a
  stationarity fallback (Arm B, Decision 9).

---

## 2. Hypotheses

### 2.1 Arm A (Holm family {H1, H2-LS, H3}, gate-passing vintages pooled)
Statistics are computed per paraphrase (pooled over vintages, formats and paths; equal weights over
rho ∈ {0.4, 0.6, 0.8, 1}) and tested with a t-test across the 12 paraphrases (11 df).
- **H1.** The one-draw (stacked) error-on-revision coefficient differs from the split-sample IV
  coefficient. The paper reports the gap, its sign and the gap predicted by the noise formula.
- **H2-LS.** The split-sample IV coefficient differs from the LS learner's on the same paths.
  (H2, the IV coefficient against zero, is reported beside it.)
- **H3.** The slope of (implied persistence − rho) on rho, rho ∈ {0, …, 1}, one-step forecasts,
  differs from zero. The population is the gate-passing models.

### 2.2 Arm B (Holm family {H4, H5})
Named framing, annual rolling valid vintages with gate substitution, COVID targets excluded,
variable intercepts, block-t over 8 time blocks.
- **H4.** The pooled split-sample IV coefficient differs from zero.
- **H5.** On pairs whose target was published on or before 2024-12-31, the coefficient under the
  2024 vintage differs from the coefficient under valid vintages (paired, block by block).

### 2.3 Power and how null results are worded
- Arm A (`docs/power.md`): with 26 vintages × 40 paths, the MDE for H2-LS is about 0.03–0.05 if
  draw noise is about one innovation s.d. and paraphrase heterogeneity tau ≤ 0.05; it cannot fall
  below 0.89·tau. The realized MDE is reported from the data.
- Arm B: the H4 MDE is at least 0.33 pre-2000 s.d. units (a smooth AR forecaster's block-t s.e.
  0.100). **A non-rejection of H4 or H5 is reported as uninformative about effects smaller than
  the realized MDE; the paper makes no "no bias" or "no leakage" claim.**

### 2.4 Inference details
- Split-sample IV in closed form (the average over every split; `ewh/estimators/eor.py`), with
  first-stage strength and Fieller sets reported for every cell.
- Per-variable Arm B results: Newey–West over quarters (4 lags).
- Holm-adjusted p-values within each family.

---

## 3. Robustness rows (pre-registered, not confirmatory)
Arm A: path-clustered inference on the full sample; rho² weights over {0.2, …, 1}; per-cell
estimates for every rho with Fieller sets; one-draw-per-prompt (draw 0) naive estimate; greedy
read; the noise ladder b(k), k ∈ {1, 2, 4, 8, 16, 32}; H3-LS.
Arm B: latest-vintage pair rule; six-vintage rolling; drop failed vintages' quarters instead of
substituting; COVID quarters included; anonymous framing; paraphrase t-test (11 df) for H4; the
1999 vintage on every quarter (constant capability) and its difference from the rolling run
(rolling and 1999 "agree" if the difference's interval contains 0 and its size is below the H4
MDE).

## 4. Secondary and exploratory analyses (no rejection claims)
- **Leakage secondaries:** framing difference-in-differences (2024 − valid gap, named minus
  anonymous, block-t); the 2009 vintage on every quarter: [2009 − 1999] on pairs the 2009 vintage
  may have seen minus the same on pairs it could not (paraphrase t-test).
- **Accuracy:** RMSE of model, SPF consensus and the AR(4) benchmark on the same pairs, including
  the 2024-versus-valid RMSE gap.
- **Human comparison:** SPF consensus and individual coefficients on the same pairs through the
  same estimators; model minus SPF.
- **Capability ladder:** corrected coefficients and noise share against parameter count, HellaSwag
  (1,000 seeded items, all models) and the Open LLM Leaderboard average (ladder only). Too few
  models for inference; descriptive only.
- Cross-template instrument (crossed block), per-variable and per-vintage results, horizons.

## 5. Competence gate (who enters the confirmatory analysis)
Run on gate seeds (`ewh/capability/gate.py`), all models, K = 32: parse rate; history sensitivity
(pairs of histories with the same last value, from rho 0.2 and 0.8); persistence (implied
persistence at rho 0, 0.5, 1); scale invariance (3x + 500). Thresholds per decision A7. Models
that fail are reported descriptively. The 1999 or 2024 vintage failing makes the test that needs
it descriptive. Arm B additionally drops pairs from vintages whose Arm B parse rate is below 95%.
Thresholds are never changed after any gate result is seen.

## 6. Timeline
Sign-off and tag → session within a week (live GPU stock permitting) → gate, Arm A, Arm B confirm
runs → adversarial review → paper.

## 7. Data already used
All raw files under `data/raw/` (SPF, RTDSM and FRED snapshot of 2026-11-11; AKLMT, BGMS and CG
replication packages; HellaSwag validation), each with a sha256 sidecar. Pilot generations under
`data/pilot/` (non-confirmatory models only).

## 8. Code
- Bound at the tag: `ewh/`, `vendor/`, `prompts/`, `configs/`, `pod/`, `pyproject.toml`, `uv.lock`.
  `git ls-tree -r pap-v1` is the authoritative list of bound blobs.
- Session: `pod/run_session.py` (with `pod/run_session.sh`, `pod/setup.sh`).
- Confirmatory: `python -m ewh.confirm.run gate|arm_a|arm_b`; dry runs of all three pass on
  synthetic generations (`tests/test_confirm_dry.py`).

## 9. Deviations
None at freeze. After the tag: `docs/deviations.md` (rule P5).
