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
