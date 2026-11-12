# Novelty check (Phase 0 gate)

**Date:** 2026-11-11. **Question:** has anyone already (C1) separated systematic
over/underreaction from forecast noise in LLM error-on-revision regressions, or (C2) run
CG/BGMS/AKLMT-style expectation tests on vintage (chronologically consistent) models against
the SPF? Handoff rule: if C1 and C2 are both published, stop.

**Verdict: neither C1 nor C2 is published. Proceed.** C3 overlaps in spirit with existing
lookahead tests, which measure contamination of accuracy, not of bias coefficients. C4 overlaps
partly with papers that compare large and small models descriptively. Details and the
overlap of every hit are below.

Scoop risk is real. The Gao–Jiang–Yan group runs the AKLMT design on LLMs and has built
vintage models (DatedGPT); the Manela group owns ChronoGPT; Jaworski and Alam already average
repeated draws for accuracy. A third-party timestamp at `pap-v1` matters (plan, decision 3).

## How the search was done

| Source | Method | Coverage note |
|---|---|---|
| arXiv | export API by id; listing pages; full text of the closest papers | full |
| OpenAlex | API keyword search, from 2024-06-01, all sources and SSRN-only (`source S4210172589`) | indexes SSRN DOIs, NBER, journals |
| Web | extended web searches (several queries) | — |
| NBER, Fed working papers, AEA program, Dataverse | direct pages (planning pre-scan) | — |
| Research Square | direct page | — |
| SSRN | site search | **blocked** by a bot check (HTTP 403); covered through OpenAlex's SSRN index instead |
| Google Scholar | site search | **blocked** by a CAPTCHA (HTTP 429); not bypassed. Covered through OpenAlex and web search |
| Semantic Scholar | API | rate-limited (HTTP 429) without a key |

Queries (combinations of): ChronoGPT; ChronoBERT; chronologically consistent; point-in-time
language model; vintage / time machine language model; DatedGPT; StoriesLM; error-on-revision;
Coibion–Gorodnichenko; overreaction / underreaction / extrapolation with LLM, ChatGPT,
language model; LLM expectations formation; LLM forecasts noise / repeated draws / sampling;
split-sample instrument; measurement error.

**Residual blind spot:** an SSRN-only paper posted in the last few weeks that OpenAlex has not
yet indexed. Re-run `uv run python scripts/novelty_openalex.py 2026-11-01` immediately before
the `pap-v1` tag and add any new hit here.

## Hits and overlap

Overlap codes: **C1** noise-corrected bias; **C2** leakage-free real-data test with vintage
models vs SPF; **C3** leakage wedge on measured bias; **C4** capability ladder.
"None" means cited for context only.

### Expectation-bias tests on language models

| Paper | Date | What it does | Overlap |
|---|---|---|---|
| Gao, Jiang, Yan, "Debiasing LLMs by Fine-tuning", arXiv:2604.02921 | v1 2026-04-03, v2 2026-05-02 (latest checked 2026-11-11) | AKLMT AR(1) design on Qwen3-32B, temperature 0, one forecast per prompt; error-on-revision b from −0.456 (rho=0) to −0.260 (rho=1); LoRA fine-tuning removes it. No noise correction; no vintage model. | Closest to C1 in design, but uncorrected. Not C2. |
| Bini, Cong, Huang, Jin, "Behavioral Economics of AI: LLM Biases and Corrections", NBER w34745 / arXiv:2602.09362 | 2026-01 / 2026-02-10 | AKLMT Experiments 1–3 on GPT-4, Claude 3 Opus, Gemini 1.5 Pro and smaller variants; implied persistence; each API call a "participant". | Partial C4 (large vs small). Not C1 or C2. |
| Chen, Green, Gulen, Zhou, "What Does ChatGPT Make of Historical Stock Returns?", arXiv:2409.11540 | 2024-09-17; draft 2025-12 (AEA 2026) | Stock-return forecasts over-extrapolate; AR(1) simulations show implied persistence too high, more so at low persistence. | None (commercial model, no correction). |
| Bybee, "Surveying Generative AI's Economic Expectations", arXiv:2305.02823 / SSRN 4430515 | 2023-05 | GPT-3.5 expectations from WSJ news vs SPF; CG regressions; consensus-like underreaction. Leakage checked only with post-cutoff articles. | Context for C2 (not vintage models). |
| Hansen, Horton, Kazinnik, Puzzello, Zarifhonarvar, "Simulating the SPF" | 2025-02 draft | GPT-4o-mini personas at temperature 1, 1999–2023. No error-on-revision regression found in the draft. | None. |
| Lin, Sun, Yan, "Simulating Macroeconomic Expectations using LLM Agents", arXiv:2505.17648 / SSRN 5265729 | v4 2025-11 | LLM agents replicate a household/expert survey experiment; heterogeneity of expectations. | None. |
| Alam, Boyle, Li, Sekhposyan, "ChatMacro", FRBSF WP 2026-04 | 2026-01 | Real-time ChatGPT inflation forecasts are inaccurate and stale out of sample. | None (motivates vintage models). |
| Yee, Koh, "Calibrating Behavioral Parameters with Large Language Models", arXiv:2602.01022 | 2026-02-01 | LLMs as instruments for loss aversion, herding, extrapolation; profile calibration. | None. |

### Noise, repeated draws and measurement error

| Paper | Date | What it does | Overlap |
|---|---|---|---|
| Jaworski, Alam, "Synthetic Consensus: Evaluating Large Language Model Forecasts with Repeated Draws", SSRN 7382959 | 2026-08-01 | 500 draws per quarterly inflation forecast, targets after the cutoff; averaging cuts forecast error by roughly 7–35%; more capable models vary less. Accuracy only (abstract and metadata seen; full text not read: SSRN blocked). | Shares C1's premise that single draws mix signal and noise, but studies accuracy, not bias coefficients. Partial C4. **Must be cited.** |
| Graham, Harvey, Jha, "CFOs Meet LLMs", arXiv:2606.13812 / SSRN 6920881 | 2026-06 | Three draws per prompt; ~4% of variance run-to-run; Spearman–Brown averaging. No CG regression. | Partial C1 premise only. |
| Das, "Over-Reading the News: How LLM Forecasters Over-Update on Headlines…", Research Square doi:10.21203/rs.3.rs-11603009/v1 | posted ~2026-11-04 | Event-probability forecasts; headlines raise forecasts 11–22 pp; repeated reads show read noise explains part of the shrinkage needed; Bayesian-consistency diagnostics. Not error-on-revision, no IV. | Adjacent to C1 (noise vs systematic over-updating), different design and outcome. Cite. |
| Canen, Enamorado, "When Predictions Become Regressors: A Split-Sample Correction…", arXiv:2608.02909 | 2026-08 | Split-sample IV for ML/LLM-predicted *regressors*. | Methodological cousin of C1's estimator; not about forecasts. Cite. |
| Kučinskas, Peters, "Measuring Under- and Overreaction in Expectation Formation", REStat 106(6) 2024 | 2024 | IV estimator robust to measurement error, human surveys. | Human precedent for C1. |
| Juodis, Kučinskas, "Quantifying Noise in Survey Expectations", QE 14(2) 2023 | 2023 | Noise is 10–30% of SPF forecast MSE. | Human precedent for C1. |
| Afrouzi, Kwon, Landier, Ma, Thesmar, QJE 138(3) 2023, §2.2 | 2023 | Notes that noise in F could make b mechanically negative. | Human precedent for C1's motivation. |

### Vintage / point-in-time models and lookahead tests

| Paper | Date | What it does | Overlap |
|---|---|---|---|
| He, Lv, Manela, Wu, "Chronologically Consistent Large Language Models", arXiv:2502.21206; SSRN 5159615; **Journal of Financial Economics (2026), doi:10.1016/j.jfineco.2026.104374** | 2025-02; JFE 2026-10-06 | ChronoBERT/ChronoGPT vintages 1999–2024; news-return predictability. | Source of our models. Not C1–C4. |
| He, Lv, Manela, Wu, "Instruction Tuning Chronologically Consistent Language Models", arXiv:2510.11677 (SSRN 5348747 "Chronologically Consistent Generative AI") | 2025-10 / 2025 | ChronoGPT-Instruct; recall of presidents/events; headline trading; greedy decoding. | Source of our models. Not C1–C4. |
| Yan, Tang, Gao, Jiang, Lu, "DatedGPT", arXiv:2603.11838 | v2 2026-07-23 | Twelve 1.3B models with annual cutoffs 2013–2024. | Alternative vintage family; no expectation tests. Scoop risk. |
| Kelly, Malamud, Schwab, Xu, "Scaling Point-in-Time Language Models", NBER w35247 / SSRN 6681860, arXiv:2607.11889 | 2026-05 | Point-in-time models up to 4B, monthly checkpoints 2013–2024. | Alternative vintage family; no expectation tests. |
| Lee et al., "PALM: Point-in-Time Adaptation for Financial Language Models", arXiv:2609.30316 | 2026-09-02 | Low-rank adaptation instead of annual pretraining; newer checkpoints not better on its task. | None. |
| Gao, Jiang, Yan, "Detecting Lookahead Bias in LLM Forecasts", arXiv:2512.23847 (v1 title "A Test of Lookahead Bias in LLM Forecasts") | v1 2025-12-29, v2 2026-06-12 | Statistical test for lookahead bias in LLM forecasts. | C3 in spirit (accuracy contamination), not bias coefficients. |
| Eliseev, Seleznev, "Fake Date Tests…", arXiv:2601.07992 / SSRN 6035916 | v2 2026-03-30 | Fake-date tests of in-sample macro forecast accuracy. | C3 in spirit. |
| Crane, Karra, Soto, "Total Recall?", FEDS 2025-044 | 2025-06 | LLM recall of GDP; errors predicted by later SPF errors. | C3 in spirit. |
| Lopez-Lira, Tang, Zhu, "The Memorization Problem: Can We Trust LLMs' Economic Forecasts?", arXiv:2504.14765 / SSRN 5217505 | 2025 | Memorization of economic data by commercial LLMs. | C3 in spirit. |
| Qian et al., "LLMs Don't Know What They Know" (FinDendro), arXiv:2611.06808 | 2026-11-04 | Benchmarks whether stated cutoffs / date prompts prevent recall. | Context for C3. |
| Liang, "Look-Ahead Bias in Financial Forecasts Generated by LLMs", SSRN 6772819; Chen et al., "Does Training on Future Data Pay?", SSRN 7486919 | 2026 | Lookahead bias in LLM financial forecasts (abstracts not available in OpenAlex; SSRN blocked). | Possible C3 context; read before the paper's literature section. |
| TiMaGPT (arXiv:2404.18543), NoLBERT (arXiv:2509.01110), StoriesLM (Sarkar 2024, SSRN), Look-Ahead-Bench (arXiv:2601.13770), SciTBERT (2026-10) | various | Other dated models and benchmarks. | None. |

## Conclusion by contribution

- **C1 (noise-corrected bias): not published.** The closest LLM papers run the uncorrected
  regression (Gao–Jiang–Yan) or average draws for accuracy (Jaworski–Alam). The idea that
  noise biases b is established for humans (AKLMT §2.2, Kučinskas–Peters, Juodis–Kučinskas).
  Our framing must credit these and position C1 as the first exact separation for LLMs,
  including the finding (plan change 2) that noise pulls b toward about −1/2 rather than
  always downward.
- **C2 (leakage-free real-data test vs SPF): not published.** No vintage-model paper runs
  CG/BGMS regressions; Bybee uses a commercial model with an in-sample cutoff.
- **C3 (leakage wedge on bias): new as a bias measure,** but lookahead tests on accuracy exist
  and must be cited as the accuracy half of the wedge.
- **C4 (capability ladder): partly anticipated** descriptively (Bini et al.; Jaworski–Alam on
  noise falling with capability). It stays descriptive, as the handoff already says.

## Re-run before the freeze (2026-11-12)
`uv run python scripts/novelty_openalex.py 2026-11-01`: no new paper overlaps C1 or C2. Closest
new items, read in full abstract: Li, Wu, Chen and Zheng, "Same Answer, Different Future: Matched
Predictions Do Not Imply Matched Belief Updating in Language Models", arXiv:2611.07913
(2026-11-05) — belief persistence in a Bernoulli task, no forecast-revision regression or noise
correction; Fu, Deng, Hong and Wang, "Timing is Everything", arXiv:2611.05781 (2026-11-04) —
temporal membership inference, context for leakage. To be re-run once more at the tag.

## Re-run at the tag (2026-11-12)
`uv run python scripts/novelty_openalex.py 2026-11-08`: 81 hits, all titles read. None concerns
forecast revisions, expectation bias, vintage or point-in-time language models, or noise
correction; no overlap with C1–C4. The verdict above stands at `pap-v1`.
