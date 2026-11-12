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

## Decision 4: Least-squares learner benchmark, Phase 1 finding (2026-11-11)
- **Finding:** simulating the LS AR(1) learner on 40 shown values (40,000 paths per rho,
  `tests/test_estimators.py::test_ls_learner_benchmark_values`): its error-on-revision
  coefficient is about −0.28 at rho = 0.2 and turns positive at higher persistence (about
  +0.07 at 0.6 and +0.10 at 1.0); its implied persistence is within about 0.01 of the true rho
  at every rho, so implied-minus-true persistence has essentially no slope in rho.
- **Consequence for the plan:** plan change 1 was half right. The LS contrast is needed for H2
  (b against zero would mislabel learning as bias, and the sign depends on rho). It is *not*
  needed for H3: the planning review's claim that Kendall bias makes the learner's H3 slope
  negative is not borne out (the cross-sectional regression of the forecast on the last value
  offsets the downward bias of the per-path slope). H3-LS stays as a cheap robustness contrast;
  the PAP's decision list should recommend H3 against zero as primary and H2-LS as primary.
- **Also:** the equal-weight pooled LS coefficient over rho in {0.2, …, 1} is close to zero,
  because the negative low-rho and positive high-rho cells cancel. Pooled H2 against zero would
  look "unbiased" for the learner while every cell is biased. Per-rho reporting is required.

## Decision 5: Estimator implementation conventions (2026-11-11)
- **Choice:** (a) one common estimation sample: units with at least 8 valid draws at both
  dates (`min_valid`, plan D6); naive_single uses the first valid draw at each date so it keeps
  the same units. (b) Influence values ψ = (a − b·d)/mean(d) at the unit level; variances use
  only the G/(G−1) cluster factor (statsmodels additionally applies (N−1)/(N−K); tests compare
  with that factor removed). (c) Two-way variance falls back to the larger one-way variance if
  the CGM sum is not positive. (d) Fieller sets reported with their kind (bounded, two rays,
  whole line). (e) Fixed-b critical values for Driscoll–Kraay are simulated
  (`fixed_b_critical`), not taken from a polynomial.
- **Reason:** (a) and (b) make the H1 difference and all pooled estimates well defined;
  (c) is the standard fix; (e) avoids an unverifiable constant.

## Decision 6: Planted forecaster used in Phase 1 tests (2026-11-11)
- **Choice:** F\*_{t−1} = rho² x_{t−1}; F\*_t = F\*_{t−1} + λ(rho x_t − F\*_{t−1}), λ = 1/(1+θ), so
  the noise-free coefficient is exactly θ for any rho > 0 (`ewh/sim/planted.py`).
- **Reason:** the planted-bias recovery test needs a known population coefficient; this
  family spans over- and underreaction with one parameter.

## Decision 7: Arm B construction and inference rules — committed before any SPF regression (2026-11-11)
Plan Task 2.0. Written after reading the BGMS and CG code (definitions only); no regression on
SPF or model data has been run. These rules bind the SPF replication (Task 2.4) and Arm B.

**Variables (14).** SPF series with real-time vintages: NGDP, RGDP, PGDP, RCONSUM, RNRESIN,
RRESINV, RGF, RGSL, HOUSING, CPI, INDPROD, UNEMP; plus TB3M and TN10Y (never revised; FRED
TB3MS and GS10 quarterly averages of monthly values). AAA dropped (Decision 3).
RTDSM codes: NOUTPUT, ROUTPUT, P, RCON, RINVBF, RINVRESID, RGF, RGSL, HSTARTS, CPI (quarterly
vintages from 1994Q3), IPT (monthly vintages; quarterly averages within each vintage), RUC.

**Target (BGMS, horizon t+3).** Growth variables and CPI: x_{t+3}/x_{t−1} − 1 (four-quarter
growth ending t+3; for CPI this equals BGMS's compounded current-quarter-through-t+3 rate).
UNEMP and rates: level in quarter t+3.

**Outcome (BGMS baseline "ql").** x_{t+3} and x_{t−1} both taken from the vintage dated t+4, the
first vintage that contains x_{t+3}; levels from that vintage (rates from FRED). Growth is
therefore never computed across two vintages, so NIPA rebasings cannot enter a ratio.

**SPF forecasts and revisions (BGMS exactly, for the human benchmark).** Growth forecast at t:
SPF level forecast for t+3 divided by the real-time value of x_{t−1} in vintage t, minus 1
(INDPROD: divided by the SPF's own t−1 column, as BGMS). Revision: that forecast minus the
t−1 survey's forecast of t+3 over its own nowcast of t−1. Levels: forecast for t+3 minus the
t−1 survey's forecast for the same quarter. CPI: compounded annualized quarterly rates.
Trimming: drop forecasts more than 5 interquartile ranges from the cross-sectional median,
per quarter and horizon. Individual regressions keep forecasters with at least 10 pairs.
Consensus = mean of individual errors and revisions by quarter.

**Model forecasts.** The prompt at survey quarter s shows the last 40 quarters of the
transformed series (four-quarter growth in percent, or the level) computed inside the RTDSM
vintage dated s, which contains data through s−1; rates show quarterly averages through s−1.
One decimal. The date-t prompt asks for the target (4 steps after the last shown value); the
date-(t−1) prompt asks for the same target (5 steps after). Revision = F_t − F_{t−1} in the
same units as the target, so model and SPF revisions measure the same quantity.

**Sample.** Forecast quarters t = 2000Q2 … 2025Q4 (the last t whose vintage t+4 is in the
2026-11-11 RTDSM snapshot, which ends at vintage 2026Q4). Primary vintage rule: both
forecasts of a pair use the ChronoGPT vintage with cutoff 31 December of year(t−1) − 1;
robustness: latest valid vintage for each forecast. Contaminated pairs for H5: target first
released on or before 2024-12-31. COVID (proposed, AJ confirms in the PAP): primary drops
pairs whose target quarter t+3 lies in 2020Q1–2021Q2; the full sample is a robustness row.

**Scaling for pooling.** Each variable's errors and revisions are divided by the standard
deviation of its outcome over target quarters 1985Q1–1999Q4, computed from the vintage dated
2000Q1 (pre-sample information only).

**Inference.** Per variable: Newey–West (Bartlett). For the replication check, BGMS's own choices
are matched (consensus: Bartlett with automatic bandwidth; individuals: two-way clustered by
forecaster and quarter); the replication tolerance applies to coefficients. For Arm B, the
pooled method is chosen by size simulation in Phase 3 between Driscoll–Kraay with fixed-b
critical values and block-t (plan D12).

**Replication tolerance (plan Task 2.4, fixed now).** Reference values: (i) BGMS Table 3 as
computed from their own processed panel (`spf_out_trim5.dta`, BGMS package) and (ii) the
Aug 2019 revision's Table 3 (`docs/references_verified.md`). Pass if, for NGDP, RGDP, PGDP, CPI,
UNEMP, TB3M and TN10Y on BGMS's sample (1968Q4–2016Q4), each consensus coefficient from our
pipeline (raw SPF and RTDSM files, 2026-11-11 snapshot) is within 0.15 of reference (i), with
the same sign as reference (i) for at least 6 of 7; and the CG inflation coefficient (PGDP
consensus, CG's sample 1969Q1–2014Q4) lies in [1.0, 1.5].

## Decision 8: SPF replication, first run and loader fix (2026-11-11)
- **First run (recorded as it happened):** verdict FAIL. All 7 headline consensus coefficients were
  within 0.15 of reference (i) with matching signs, but the CG inflation coefficient (our PGDP
  consensus on 1969Q1–2014Q4) was 1.522, outside [1.0, 1.5]. RGDP consensus was 0.59 against
  0.44.
- **Diagnosis:** a row-by-row merge with BGMS's panel showed our outcomes, forecasts and
  revisions equal theirs exactly wherever both exist; the only difference was two missing
  quarters (t = 1995Q1 and 1996Q1). The 1996Q1 NIPA vintage lacks the 1995Q4 observation
  (delayed release after the 1995–96 shutdown); BGMS take it from the 1996Q2 vintage, our loader
  did not.
- **Fix:** `realtime.value` now takes a vintage's missing latest observation from the next
  vintage that has it (at most two quarters later), never older observations. This is a data
  rule, not a tolerance change; the tolerance and the CG band are unchanged.
- **Second diagnosis and fixes (same day):** after the fix, T and N matched BGMS exactly but
  t = 1996Q1 still differed: the 1995Q4 value exists only in the 1996Q2 vintage, which is on the
  new chain-weighted base (January 1996 comprehensive revision) while the 1996Q1 survey's
  forecasts are on the old base. (a) `realtime.outcome` now reads numerator and denominator from
  one effective vintage (the first that holds x_{t+3}), which is the rebasing rule of Decision 7
  applied consistently. (b) In replication mode only, the SPF forecast denominator for a survey
  whose vintage lacks x_{t−1} uses BGMS's extrapolation rescaled by their base-change ratio.
  Note for the paper: this single quarter moves the consensus RGDP coefficient by about 0.14,
  which shows how fragile consensus CG coefficients are to one quarter with mis-based revisions.
- **Final result:** after (a) and (b) and matching BGMS's spline fill for NGDP in 1996Q1, our
  pipeline reproduces reference (i) to two decimals for all 14 series (consensus and individual),
  and the CG coefficient is 1.41 on both our pipeline and BGMS's panel (CG's own processed data
  give 1.19). Verdict PASS under the unchanged Decision 7 tolerance. `docs/spf_replication.md`
  is the generated record. The first-run FAIL is disclosed here and will be in the PAP.

## Decision 9: Real-time AR benchmark must be stationary (2026-11-11)
- **Finding:** in the Arm B power check, real-time AR(4) fits on histories containing the 2020
  collapse are explosive and give revisions of thousands of standard deviations (e.g. UNEMP,
  t = 2020Q4).
- **Choice:** the real-time AR benchmark (plan D12, used for RMSE comparisons and the power
  check) fits an AR(4) with intercept to the 40 shown values; if the fit is not stationary it
  falls back to an AR(1) with coefficient capped at 0.99. Fixed before any model output.

## Decision 10: Proposals for the PAP from the Phase 3 power work (2026-11-11)
Proposals only; AJ decides in Phase 4. Evidence in `docs/power.md`.
- **Template inference (plan D8):** per-paraphrase estimates and a t-test across the six
  paraphrases (5 df) as primary. It is the only candidate that holds its size when the true
  coefficient varies across wordings; clustering by path over-rejects (6–9%) and the wild
  bootstrap over-rejects with no heterogeneity (5–8%).
- **Pooled Arm A estimand (plan change 3, refined):** equal weights over rho in {0.4, 0.6, 0.8,
  1} as primary; fixed weights proportional to rho^2 over {0.2, …, 1} as robustness; rho = 0
  and 0.2 per cell with Fieller sets. Reason: for near-rational or underreacting forecasters the
  rho = 0.2 cell's instrument is weak once per-draw noise is about one innovation s.d. or more,
  and equal weights over {0.2, …, 1} inherit it.
- **K = 32 draws** per prompt (plus one greedy read): clearly better than 16 at high noise.
- **Arm B pooled inference (plan D12):** block-t over 8 time blocks (7 df) as primary;
  Driscoll–Kraay over-rejects (13%) even with fixed-b critical values on 97 overlapping quarters.
- **H4 MDE** is at least 0.33 pre-2000 s.d. units, so the pre-registered "uninformative below
  the MDE" wording will apply to any non-rejection.
- **MDE target 0.05 (plan D13)** is attainable only if the across-paraphrase s.d. of the true
  coefficient (tau) is about 0.025 or less. Options for AJ: accept a larger MDE, or add
  paraphrases before the freeze (doubling to 12 paraphrases lowers the tau floor from 1.43 tau to 0.89 tau, about 38%);
  no ChronoGPT output exists yet, so adding templates now is still allowed.

## Decision 11: Sampler accepted; proposed session preflight rule (2026-11-12)
- **Finding:** on full weights (pilot 2) the cached sampler agrees with the authors' code to bf16
  tolerance (teacher-forced top-1 agreement 316/322 at margins above 0.1; sampled-number KS
  p = 0.95). Exact greedy identity is not attainable in bf16.
- **Proposed preflight for every instruct vintage in the session (for the PAP):** on 50 pilot-
  table prompts, teacher-forced top-1 agreement at least 95% of steps with authors' margin above
  0.1, and prompt-pass max KL below 0.1; a vintage that fails is not generated and is reported.

## Decision 12: Twelve paraphrases (templates v2) — AJ's decision (2026-11-12)
- **Decision (AJ, 2026-11-12):** double the paraphrases from 6 to 12 before the freeze, giving
  24 templates (12 paraphrases x 2 number formats) in both arms. P1–P6 are unchanged; P7–P12 were
  added. Level-series descriptions now give units in parentheses ("(percent)") so no named
  sentence breaks; this edits only the Arm B named framing of UNEMP, TB3M and TN10Y.
- **Reason:** with template as a random factor, the Arm A MDE floor from paraphrase
  heterogeneity falls from 1.43·tau to 0.89·tau (t-test across paraphrases, 11 df instead of 5);
  at tau = 0.05 the floor is 0.044 instead of 0.071 (`docs/power.md`). Compute is not binding.
- **Disclosure for the PAP:** templates v1 (sha 44d2e514…) were committed before any ChronoGPT
  output. Before v2, the only ChronoGPT-family output was pilot 2 on the non-instruct base model
  with v1 templates, from which only sampler-equivalence, parse-rate and throughput statistics
  were computed (no forecast or gate statistic), plus pilot 1 on Qwen2-0.5B (v1 templates, rho 0.5,
  pilot seeds). No outcome informed the new wording. Templates v2 sha256 is recorded in
  `configs/generation.yaml`.
- **Consequences:** Arm B has 137,088 prompts per full run (was 68,544); Arm A doubles per path;
  the size check with 12 paraphrases still shows the paraphrase t-test holding its size.

## Decision 13: Pairs with no published outcome get no prompts (2026-11-12)
- **Finding (dry run of the confirm scripts):** UNEMP, t = 2025Q1 has no outcome: its target
  quarter 2025Q4 lacks a quarterly average because the October 2025 unemployment rate was never
  published (government shutdown). A NaN outcome made within-variable demeaning NaN for the whole
  variable and the block t-test then dropped blocks silently.
- **Rule:** `build_arm_b` skips any pair whose outcome is not published in the 2026-11-11 snapshot.
  Block and paraphrase t-tests report the number of non-finite group estimates (`n_nonfinite`)
  instead of dropping them silently. Fixed before any model output.

## Decision 14: Seed-tag hashing bug fixed before any confirmatory generation (2026-11-12)
- **Finding (dry run of the Arm A confirm script):** `ewh.sim.ar1._tag` turned a string tag into
  an integer by keeping only its first 8 bytes, so tags such as "chrono-2000" and "chrono-2001"
  (and "gate-hist-a" / "gate-hist-b") gave the same random stream. Arm A main-block paths were
  therefore shared across vintages (2,304 distinct prompts instead of 29,952 in the dry run),
  breaking the nested design (plan change 4).
- **Fix:** tags are hashed in full with blake2b (64 bits). Regression tests added. This changes
  every stream keyed by a string tag. No confirmatory prompt had been generated. The pilots and the
  Phase 1-3 test fixtures used the old streams; the pilots are logged with the tables they used and
  no pilot statistic depends on which paths were drawn.
