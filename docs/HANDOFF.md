# HANDOFF: Expectations Without Hindsight

**Working title:** *Expectations Without Hindsight: Noise, Capability and Bias in Language-Model Forecasts*
**Author:** Ajai Upadhyaya (solo). **Prepared:** 2026-11-11. **For:** Claude Code.
**Output:** a pre-registered working paper (PDF), a one-command replication repo, and a short web summary for ajaiupadhyaya.com.

Read this whole file before writing code. Sections 3 and 4 are the reason the project exists; do not simplify them away.

---

## 1. The question in plain terms

Several papers report that language models form expectations "like people": their forecasts over-extrapolate, and regressions of forecast errors on forecast revisions look like the ones estimated on human surveys. Three things could produce that result without any human-like belief formation:

1. **Hindsight (leakage).** The model was trained on text written after the dates it is asked to forecast.
2. **Noise.** Random variation in a forecast mechanically produces a *negative* error-on-revision coefficient, which is read as "overreaction" (Section 3).
3. **Weakness.** A small model that just echoes the last number it saw also looks like it overreacts.

This project measures expectation bias in language models after removing each of the three, and reports how much of the published pattern each one accounts for.

## 2. What is and is not new (be honest about this in the paper)

Already done by others, and to be cited, not claimed:

- LLM macro expectations compared with the SPF, including error-on-revision regressions: Bybee (2023), arXiv:2305.02823. Uses a commercial model with a training cutoff inside the sample.
- LLM over-extrapolation that is stronger for less persistent series: Chen, Green, Gulen and Zhou (2025).
- The Afrouzi et al. AR(1) forecasting experiment adapted to an open LLM (Qwen): Gao, Jiang and Yan (2026), arXiv:2604.02921.
- Tests showing lookahead bias in LLM macro forecasts: Gao, Jiang and Yan (2026), arXiv:2512.23847; Eliseev and Seleznev (2026), arXiv:2601.07992.
- Chronologically consistent open-weight models: He, Lv, Manela and Wu (2025), arXiv:2502.21206 and arXiv:2510.11677.

Intended contributions:

- **C1. Noise-corrected bias.** Because we control sampling, we can separate systematic over/underreaction from forecast noise exactly (repeated draws, split-sample IV). Human-survey papers cannot do this.
- **C2. Leakage-free real-data test.** Error-on-revision regressions on real-time macro data using vintage models that cannot have seen the outcome, compared with SPF consensus and individual forecasts on the same quarters.
- **C3. The leakage wedge.** The same regressions with a model that *has* seen the outcome, to quantify what hindsight does to measured bias and accuracy.
- **C4. Capability ladder.** Whether the corrected bias shrinks as model capability rises.

**Phase 0 novelty gate (do this first, about 2 hours, no GPU).** Search arXiv, SSRN, NBER and Google Scholar for any paper that already runs error-on-revision or Afrouzi-style tests on ChronoGPT/ChronoBERT/StoriesLM or other vintage models, or that applies a measurement-error correction to LLM forecast regressions. Write `docs/novelty_check.md` with every hit, its date, and which of C1 to C4 it overlaps. If C1 and C2 are both already published, stop and report to AJ before doing anything else.

## 3. The main risk and how the design removes it

AJ's condition for doing this project: the weak-model problem must not be able to skew the results. Treat the following as binding design requirements.

### 3.1 Why noise fakes overreaction

Let the model's systematic forecast be F\*, and an observed draw be F = F\* + u, with u independent across draws. The individual-level regression is

    (x_{t+h} - F_t)  =  a + b (F_t - F_{t-1}) + e

The draw noise u_t enters the revision with a plus sign and the error with a minus sign, so

    plim b_naive = [ Cov(x - F*_t, rev*) - Var(u) ] / [ Var(rev*) + 2 Var(u) ]

A perfectly rational but noisy forecaster (true b = 0) gets b_naive < 0. A weak model is a noisy model, so the naive regression cannot distinguish "overreacts" from "is noisy". Derive this formally in the paper and confirm it by Monte Carlo before any model is run.

### 3.2 Required estimators (report all three side by side, everywhere)

1. **Naive:** one draw per prompt (what the literature mostly reports).
2. **Mean of K:** average K independent draws per prompt; noise variance falls as 1/K. Report the coefficient as a function of K (the "noise ladder" figure: K = 1, 2, 4, 8, 16, ...).
3. **Split-sample IV (primary):** split each prompt's draws into disjoint halves A and B. Regress the error built from A on the revision built from A, instrumenting with the revision built from B. Noise in A is independent of B, so the estimate is consistent for the noise-free coefficient. Report first-stage strength; average over many random splits or use all cross-fold pairs, fixed in the plan.

Also report a deterministic read (greedy decoding) as a secondary, noting it is the mode, not the mean.

### 3.3 Capability controls

- **Competence gate, fixed before outcomes are read.** A model enters confirmatory analysis only if it passes checks unrelated to the hypotheses: (a) parse rate at or above a stated threshold; (b) history sensitivity: with the last observation held fixed, the forecast must respond to the rest of the history, and implied persistence must rise with true persistence across conditions (rules out "echo the last number"); (c) scale invariance: shifting and rescaling the series shifts and rescales the forecast. Thresholds go in the pre-analysis plan. Models that fail are reported descriptively and excluded from tests.
- **Capability ladder in the synthetic arm.** Simulated series are generated fresh from our seed, so leakage is impossible and any model is admissible. Run the ChronoGPT-Instruct models alongside modern open instruct models at several sizes (candidates: Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B, 14B; Llama-3.2 1B, 3B; confirm current availability and licences). Plot corrected bias against size and against a standard capability score. This shows whether a 1.55B vintage model behaves like its size class and whether bias fades with capability.
- **Feasible rational benchmarks.** Bias is measured against (a) the known-parameter rational forecast and (b) a least-squares AR(1) learner given exactly the history in the prompt. Also report random-walk and unconditional-mean forecasters. All benchmarks are pushed through the identical estimation code, including artificially added noise, as a validity check.
- **Constant-capability check in the real-data arm.** Vintages differ in training tokens, so capability drifts over the sample. Run (i) rolling vintages and (ii) the 1999 vintage alone on every quarter (leakage-free throughout, capability fixed). Agreement between (i) and (ii) rules out capability drift as the driver.
- **Same-model revisions.** A revision F_t - F_{t-1} must be computed from one model. If the vintage switches between t-1 and t, the revision mixes a belief update with a model change. Primary rule: both forecasts in a revision pair use the vintage valid at t-1. Robustness: latest valid vintage for each.
- **Prompt variation as a cluster.** Pre-register a fixed set of prompt templates (at least 5 paraphrases, 2 number formats). Template is a random factor, never a tuned choice. No template is dropped after outcomes are seen.

## 4. Design

### 4.1 Arm A: controlled forecasting experiment (synthetic)

- Follow Afrouzi, Kwon, Landier, Ma and Thesmar (2023, QJE 138(3), doi:10.1093/qje/qjad009): AR(1) processes with persistence in {0, 0.2, 0.4, 0.6, 0.8, 1.0}, a history of past realizations shown, one- and two-step-ahead forecasts. **Read the paper and its online appendix and copy the exact process parameters (mean, innovation sd, history length, rounds); do not rely on this summary.**
- Use integer-valued series to limit tokenizer artefacts (the ChronoGPT tokenizer is GPT-2 BPE). Check how each model tokenizes the number range used and document it.
- Each trial: a simulated path, a prompt template, a model, K draws at each of two consecutive dates (so revisions exist).
- Outcomes: error-on-revision coefficient (three estimators), implied persistence (regress forecast on last observation, compare with true rho), and the noise share Var(u)/Var(F).
- Human comparison: Afrouzi et al.'s replication data if publicly posted; otherwise their published coefficients.

### 4.2 Arm B: real-time macro forecasts with vintage models

- **Models:** `manelalab/chrono-gpt-instruct-v1-YYYY1231`, annual cutoffs 1999 to 2024, about 1.55B parameters each, MIT licence. Verify the exact repo ids, prompt format and loader from the model cards (they use a custom `ChronoGPT` class, not `AutoModel`). Record the commit hash of every model used.
- **Sample:** 2000Q1 to the latest quarter with realized data. A forecast dated in year Y uses a cutoff no later than Dec 31 of Y-1.
- **Data (public only):** Philadelphia Fed Survey of Professional Forecasters (consensus and individual files); Philadelphia Fed Real-Time Data Set for Macroeconomists (the history a forecaster could see at each survey date, and first-release realizations); FRED/ALFRED where needed. Variable list: start from the set in Bordalo, Gennaioli, Ma and Shleifer (2020, AER, doi:10.1257/aer.20181219) and keep those with both SPF forecasts and real-time vintages. Check each source's terms of use and record them.
- **Prompt information set:** only the real-time history available at the SPF survey deadline for that quarter. Two pre-registered framings: *named* (variable name and date given) and *anonymous* (numbers only), to separate what the model knows about the economy from pattern extrapolation.
- **Outcomes:** the Coibion and Gorodnichenko (2015) / BGMS error-on-revision regression at the BGMS horizon, per variable and pooled; forecast accuracy against SPF and a real-time AR benchmark.
- **Human comparison:** the identical code on SPF consensus and SPF individual forecasts, same quarters and variables; test the difference between model and human coefficients.
- **Leakage wedge:** rerun everything with the 2024 vintage on all quarters (contaminated before 2025) and report the change in coefficient and in RMSE relative to the valid-vintage run.

### 4.3 Hypotheses (finalize wording in the plan; sign conventions: b < 0 overreaction, b > 0 underreaction)

Confirmatory, Arm A (one Holm family):
- **H1.** The naive coefficient is more negative than the split-sample IV coefficient (noise inflates measured overreaction).
- **H2.** The split-sample IV coefficient differs from zero, pooled over gate-passing ChronoGPT models (two-sided; direction reported).
- **H3.** Corrected overreaction, measured as implied minus true persistence, is larger at lower persistence.

Confirmatory, Arm B (separate Holm family):
- **H4.** The pooled split-sample IV coefficient for valid-vintage forecasts differs from zero (two-sided).
- **H5.** The coefficient under the contaminated 2024 vintage differs from the valid-vintage coefficient (leakage wedge).

Descriptive or exploratory (labelled as such, no rejection claims): the capability ladder (too few models for inference; say so), model-versus-SPF coefficient gaps by variable, named versus anonymous framing, horizon effects.

Every confirmatory result is reported whether or not it rejects, with interval, two-sided p-value and minimum detectable effect.

### 4.4 Inference

- Arm A: cluster by simulated path and by template (two-way); model fixed effects. Paths are cheap, so set the number of paths from the power analysis, not from convenience.
- Arm B: errors overlap across horizons and co-move across variables within a quarter. Use Driscoll-Kraay or quarter-clustered errors with an appropriate lag; check what BGMS use and match it for comparability. Verify size by simulation on this panel's design before choosing, as was done in AJ's earlier papers.
- **Power first.** Before freezing, simulate each confirmatory test under the planned sample and report the MDE. If an Arm B MDE is too large to be informative, the plan must say in advance how that result will be worded. Arm A must be sized so its MDE is small; that is the point of having it.

## 5. Protocol (same discipline as AJ's earlier pre-registered paper)

1. Write `docs/PAP.md`: every outcome, regressor, sample, estimator, inference method, decision rule, gate threshold, template list, seeds, robustness rows. Include a decision register and a disclosures section listing everything seen before the freeze.
2. Pilot only on pilot seeds, a persistence value outside the confirmatory set, and a model outside the confirmatory set. Pilots tune parsing and throughput, nothing else. Log them.
3. Run the full pipeline on a stand-in model (for example GPT-2 small on CPU) and on synthetic "forecasters" with known bias and known noise. The pipeline must recover the planted bias under all three estimators. This is a required test, not optional.
4. Freeze: annotated git tag `pap-v1`. Confirmatory scripts verify the tag, a clean tree and input hashes, write an attempt marker before reading outcomes, and refuse to rerun.
5. After the tag, any change is a logged deviation in `docs/deviations.md` with a "results seen" field. A deviation made after seeing a confirmatory result demotes that hypothesis to exploratory; the pre-registered result is still reported as primary.
6. Before writing up, a separate Claude instance with access only to code, result tables and the plan's identification section writes an adversarial review. Address each point in `docs/review_01_checks.md`.
7. No number in the paper is typed by hand. Every reference is checked against its source (title, authors, year, venue, DOI). `make all` rebuilds data, tables, figures and the PDF from archived raw files with source URL, retrieval time and hash sidecars.
8. Include an AI-use statement, data and code availability, licences, and a limitations section that leads with what the design cannot establish.

## 6. Compute plan and budget

**Hard limit: about $50 of RunPod credit, roughly 6 H100-hours. Plan for 4.5 hours of GPU work and keep 1.5 in reserve. Ask AJ before exceeding it.**

- Build and test everything on CPU first. The GPU pod is started only when the dry run passes end to end.
- **Downloads are the hidden cost.** Each ChronoGPT model is about 7.4 GB; 26 vintages is about 190 GB before the ladder models. Pre-download to a RunPod network volume from a cheap CPU pod (check current pricing and volume fees), so GPU time is not spent downloading. If storage or time is tight, the vintage priority order is: 1999, 2024, then 2004, 2009, 2014, 2019, then the rest. State in the plan which set is confirmatory.
- First 10 GPU minutes: measure prompts per second per model at the planned batch size and context length (context limit is 1,792 tokens). Then set K, paths and templates from the power analysis subject to the time budget. Write the resulting allocation to `docs/compute_budget.md` before the confirmatory run.
- Checkpoint after every model and every shard; runs must resume without redoing work. Save every raw generation (prompt hash, model, commit, seed, draw index, raw text, parsed value) to parquet. Raw generations are the archived data.
- Fix seeds; record library versions, GPU type and driver. Sampling temperature and top-p are fixed in the plan.
- Order of the GPU session: (1) throughput test, (2) competence gate, (3) Arm A ChronoGPT, (4) Arm B valid vintages, (5) Arm B contaminated vintage, (6) Arm A ladder models, largest last. If time runs out, the ladder is what gets cut, since it is descriptive.
- Shut the pod down when done and confirm it is stopped.

## 7. Repo layout

    expectations-without-hindsight/
      Makefile                  # make all
      docs/                     # PAP.md, novelty_check.md, decisions.md, assumptions.md,
                                # deviations.md, compute_budget.md, review_01*.md
      data/raw/                 # archived downloads + MANIFEST.json (url, time, sha256)
      data/generations/         # parquet of raw model outputs
      ewh/
        sim/                    # AR(1) generator, rational and LS benchmarks
        data/                   # SPF, RTDSM loaders; real-time information sets
        prompts/                # templates, formatting, parsing
        models/                 # loaders, batched sampling, checkpointing
        estimators/             # naive, mean-of-K, split-sample IV, implied persistence
        inference/              # clustering, HAC, power simulations
        confirm/                # one script per confirmatory family, tag-bound
      tests/                    # includes planted-bias recovery test
      paper/                    # source, figures, tables (all generated)
      web/                      # short summary page for the site

## 8. Phases and stop points

| Phase | Work | GPU | Stop and report to AJ if |
|---|---|---|---|
| 0 | Novelty check, read the five core papers, verify data access and licences | no | C1 and C2 already published; any data source not usable publicly |
| 1 | Simulator, estimators, Monte Carlo of the noise result, planted-bias tests | no | split-sample IV fails to recover planted bias |
| 2 | Data loaders, real-time information sets, SPF replication of published CG and BGMS coefficients | no | cannot reproduce the published human coefficients to a reasonable tolerance |
| 3 | Prompts, parsing, CPU dry run with stand-in model, power analysis | no | Arm A MDE is not small, or budget cannot cover the confirmatory set |
| 4 | Write PAP, AJ approves, tag `pap-v1` | no | always: AJ signs off before the tag |
| 5 | Single GPU session per Section 6 | yes | fewer than half the vintages pass the competence gate |
| 6 | Estimation, robustness, adversarial review, checks | no | |
| 7 | Paper, replication package, web summary | no | |

If most ChronoGPT vintages fail the gate, that is itself a finding about using small vintage models as synthetic forecasters, and the paper is reframed around it rather than loosened to let them through. Do not lower a gate threshold after seeing who fails.

## 9. Decisions AJ still owns

- Final wording of H1 to H5 and the gate thresholds (Phase 4).
- Whether the anonymous or the named framing is primary in Arm B. Default proposed here: named is primary, anonymous is a pre-registered secondary.
- Whether to add credit if the power analysis shows the full vintage set does not fit in 4.5 hours.
- Title.

## 10. Core references to read and verify first

- Coibion and Gorodnichenko (2015), "Information Rigidity and the Expectations Formation Process", AER.
- Bordalo, Gennaioli, Ma and Shleifer (2020), "Overreaction in Macroeconomic Expectations", AER, doi:10.1257/aer.20181219.
- Afrouzi, Kwon, Landier, Ma and Thesmar (2023), "Overreaction in Expectations: Evidence and Theory", QJE, doi:10.1093/qje/qjad009.
- Kucinskas and Peters, "Measuring Biases in Expectation Formation" (IV approach robust to measurement error); Juodis and Kucinskas (2023), "Quantifying Noise in Survey Expectations", Quantitative Economics.
- He, Lv, Manela and Wu (2025), arXiv:2502.21206 and arXiv:2510.11677.
- Bybee (2023), arXiv:2305.02823; Chen, Green, Gulen and Zhou (2025); Gao, Jiang and Yan (2026), arXiv:2604.02921 and arXiv:2512.23847; Eliseev and Seleznev (2026), arXiv:2601.07992.

The descriptions of these papers in this file come from abstracts and search results. Confirm every claim against the paper itself before it appears in the draft.
