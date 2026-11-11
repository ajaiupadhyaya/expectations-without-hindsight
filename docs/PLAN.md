# Expectations Without Hindsight: execution plan

**Spec:** `/Users/ajaiupadhyaya/Downloads/HANDOFF-expectations-without-hindsight.md`. First commit copies it to `docs/HANDOFF.md` and this plan to `docs/PLAN.md`; the executor reads both. Where they disagree the handoff wins, except for the numbered items under "What planning changed", which are deliberate.

**Goal:** a pre-registered working paper, a one-command replication repo and a web summary that measure expectation bias in language-model forecasts after removing leakage, forecast noise and weak-model artefacts.

**Shape of the work:** a Python package `ewh` built and tested on CPU; a pre-analysis plan frozen by an annotated tag `pap-v1`; a short GPU pilot before the tag on a non-confirmatory model; one tag-bound GPU session that only produces raw generations; tag-bound confirmatory scripts that run once; a LaTeX paper whose every number is generated.

**Stack:** Python 3.12, uv, hatchling, pytest, ruff, numpy/pandas/pyarrow/scipy, torch (stable), tiktoken, transformers (ladder and stand-in models only), LaTeX via latexmk.

---

## Context

The handoff asks whether language models "form expectations like people" once three artefacts are removed: training-data leakage, sampling noise in the error-on-revision regression, and weak models that echo the last number. Planning verified the facts the handoff marked unverified, then had the draft design checked by an independent reviewer. Both steps changed the design. Nothing has been created yet: no repo, no pod, no downloads.

### What planning verified (2026-11-11)

| Item | Finding |
|---|---|
| Novelty | No paper found that noise-corrects an LLM error-on-revision regression (C1) or runs CG/BGMS/AKLMT tests on vintage models (C2). SSRN and Google Scholar could not be searched by tool; Phase 0 finishes the check in a browser. |
| Closest work | Gao, Jiang, Yan (arXiv:2604.02921): AKLMT design on Qwen3-32B, **temperature 0, one read per prompt**, b from -0.46 (rho=0) to -0.26 (rho=1). Bini, Cong, Huang, Jin (NBER w34745): implied persistence only. Neither corrects for noise. |
| Scoop risk | The DatedGPT group (arXiv:2603.11838) has vintage models and an AKLMT pipeline; the Manela lab owns ChronoGPT. |
| ChronoGPT-Instruct | All 26 vintages exist (`manelalab/chrono-gpt-instruct-v1-YYYY1231`, 1999 to 2024), open, MIT, 7.43 GB each in fp32, 52 layers, dim 1536, context 1,792, tokenizer `tiktoken` gpt2. No v2. |
| ChronoGPT code | Custom class in `ChronoGPT_instruct.py`. Its `generate` takes one prompt, has no working KV cache, and computes logits for every position. The forward hard-casts activations to bf16. Zero-initialised output layers at construction. Needs torch 2.4 or later (`F.rms_norm`); the pinned nightly is not required. |
| Instruct tuning data | The same set for every vintage, screened to "pre-2000" by a GPT-4.1 classifier, not by timestamps. |
| AKLMT | x_{t+1} = rho x_t + e, mean 0, sd(e) 20, rho in {0, .2, .4, .6, .8, 1}, 40 past values shown, forecasts of t+1 and t+2. Replication data public, CC0: doi:10.7910/DVN/3Q6BHC. |
| BGMS | Target t+3; growth from t-1 to t+3 for macro series, level for rates; real-time outcomes; consensus SE Newey-West, individual SE two-way by forecaster and time. Package: openICPSR doi:10.3886/E118422V1 (access unverified). |
| Data | SPF and RTDSM files download from philadelphiafed.org without login. Moody's AAA/BAA is not in the public SPF files. TB3M and TN10Y have no RTDSM vintages because they are never revised. |
| RunPod | H100 SXM $3.99/h, A100 80GB $1.79/h (Secure). Network volume $0.07/GB/month, one data center, Secure only. Today no H100 is in stock where network volumes are supported; A100 80GB is. Account: no pods or volumes; balance not readable by tool. |
| This Mac | uv, LaTeX, gh, an HF token. `runpodctl` not installed. 121 GB free, 16 GB RAM. |

### What planning changed relative to the handoff

Each item is a deliberate departure or addition. Items 1 to 6 change what the hypotheses mean and go to AJ as decisions at the top of the PAP.

1. **"Bias differs from zero" needs a feasible benchmark.** A least-squares learner fitted to the 40 shown values is unbiased in any sensible meaning, yet it gives b near -1 at rho = 0 and about -0.35 at rho = 0.2, and its implied-minus-true persistence slopes down in rho (Kendall bias). H2 and H3 against zero would therefore reject for a forecaster with no behavioural bias at all. Each is reported against zero, as the handoff words it, and as a contrast against the least-squares learner on the same paths. Recommendation: the contrast carries the word "bias" in the paper.
2. **Noise does not always push b negative.** With independent noise variances s_t² and s_{t-1}², plim b_naive = (C - s_t²) / (V + s_t² + s_{t-1}²), so noise pulls b toward -s_t²/(s_t² + s_{t-1}²), about -1/2, from either side. A rational forecaster is pulled down; a strong overreactor (b* below -1/2) is pulled up. H1 is tested two-sided and the paper states the attraction point. The handoff's formula is the equal-variance special case.
3. **The pooled coefficient is defined per rho cell with equal weights.** A regression pooled over rho weights cells by revision variance, which puts about 45% of the weight on rho = 1 and almost none on low rho, and the naive and corrected estimators would use different weights. rho = 0 is excluded from b-based hypotheses in advance (the rational revision there is identically zero) and reported separately.
4. **Paths are nested, not crossed.** The shock to x_{t+1} is shared by every template and model that sees a path, so the effective sample is the number of distinct paths. Each model x template unit gets its own paths. A small crossed block serves the cross-template instrument.
5. **H5 alone cannot separate leakage from capability.** Two pre-registered secondaries are added: the named-minus-anonymous difference in the 2024-versus-valid gap, and a mid vintage (2009) run on every quarter to give a difference-in-differences against the 1999 vintage across the 2009 cutoff.
6. **COVID quarters need a rule fixed in advance.** BGMS end in 2016; this sample runs to 2025.
7. **A custom batched, cached sampler for ChronoGPT is required.** With the authors' `generate` the sample would take hundreds of GPU-hours. The sampler must be shown equivalent to the authors' forward pass before it touches a confirmatory model.
8. **The split-sample IV is implemented in closed form.** Averaging over all splits has an exact expression in per-prompt means and draw variances. The handoff allows "all cross-fold pairs". Random-split code exists only to test the closed form.
9. **A cross-template instrument is a pre-registered secondary.** Gao et al. find b < 0 at temperature 0, where sampling noise is zero; prompt-wording noise is the candidate explanation this tests.
10. **Naive is reported two ways:** one draw per prompt (draw 0), as the handoff specifies, and all single draws stacked (same limit, less variance), which the H1 test uses.
11. **GPU order is by vintage, not by arm.** Each vintage is loaded once and runs its gate, Arm A and Arm B rows. The six priority vintages finish first so a complete confirmatory dataset exists early.
12. **Generation is bound to the tag** (seeds, K, shard list, model revisions, pod scripts), not only estimation.
13. **Arm B starts at 2000Q2.** The 2000Q1 pair would need a 1998 vintage.
14. **Variables:** drop AAA (proprietary); keep TB3M and TN10Y from FRED as never-revised series.
15. **Citations to fix:** Kucinskas and Peters was published as "Measuring Under- and Overreaction in Expectation Formation", REStat 106(6), 2024; Chen, Green, Gulen, Zhou is arXiv:2409.11540 (Sep 2024); arXiv:2512.23847 changed title between versions.
16. **"Leakage-free" is a claim about pretraining.** The instruction-tuning screen is a classifier. This leads the limitations section.

---

## Global constraints

- Handoff Sections 3 and 4 are binding. No gate threshold, template, estimator or sample rule changes after outcomes are seen.
- **$50 of RunPod credit in total, storage included, is the hard limit** (AJ, 2026-11-11). Ask before exceeding it. Get AJ's explicit go, quoting the hourly price, before creating any pod or volume.
- Build and test on CPU first. The confirmatory pod starts only after the dry run passes end to end.
- Pilots use only pilot seeds, rho = 0.5 and models outside every confirmatory set. Each pilot is logged in `docs/pilot_log.md`. On the ChronoGPT base model the only statistics computed are tokens per second, parse rate and sampler equivalence; no forecast or gate statistic.
- Templates are written and hashed before any ChronoGPT-family model generates anything.
- No number in the paper is typed by hand. Every reference is checked against its source.
- `make all` rebuilds from archived raw files offline. Raw files carry URL, retrieval time and sha256.
- Public data only; record each source's terms.
- uv for everything. Git author `ajaiupadhyaya@users.noreply.github.com`, set before the first commit. Commits end with the Claude co-author line.
- On this Mac: at most 4 worker processes, `nice`, resumable, battery guard (pattern: `~/Documents/chess-tilt/src/chess_tilt/power_guard.py`).
- The handoff's Section 8 stop points are real stops: report to AJ and wait, with the pod stopped.
- End every working session with phases done / total, what is left, and the next task.

## Review focus (failure modes no single task owns)

1. **A draw that parses to a wild number or to the period index.** Expected: invalid by the fixed rule, never averaged in. Task 3.2.
2. **A prompt whose draws are all identical.** Expected: finite estimates, noise share 0. Task 1.2.
3. **A revision pair that straddles a vintage change.** Expected: the pair builder refuses under the primary rule. Task 2.3.
4. **A GPU run resumed after a crash mid-shard.** Expected: the shard is redone under its pre-committed seed; no duplicate or missing rows; the ledger shows it. Task 3.4.
5. **A real-time history or outcome that mixes vintages or base years.** Expected: every prompt history rebuilds from one vintage file, and no growth rate spans a rebasing. Task 2.2.

---

## Design decisions (defaults; AJ finalizes in the PAP)

**D1. One number per generation.** The date-(t-1) prompt asks for x_{t+1} (two steps); the date-t prompt asks for x_{t+1} (one step). Draws at the two dates are independent.

**D2. Arm A sample structure.** AKLMT's process exactly (mean 0, sd 20), integers, levels. One revision pair per path: 40 values shown at t-1, 41 at t. Main block: a fresh set of paths for every model x template unit, from a separate seed stream per rho. Crossed block: a smaller set of paths shown under all 12 templates, for the 1999 and 2024 vintages. Clustering is by global path id everywhere.

**D3. Sampling.** Temperature 1.0, no top-k, no top-p, `max_new_tokens` 12; plus one greedy read per prompt. Decoded text is stripped of leading whitespace and cut at the first newline. Random numbers come from a counter-based stream keyed on a hash of (master seed, model, prompt_id, draw, step), so no two prompts share uniforms. Every sampling argument is passed explicitly to every backend. ChronoGPT runs in the precision of the authors' tutorial.

**D4. Per-prompt statistics.** For prompt pair i with valid draws at each date: ebar = x - mean(f_t); rbar = mean(f_t) - mean(f_{t-1}); s_t², s_{t-1}² the draw variances; K_t, K_{t-1} the valid counts. Noise-free moments: N_i = ebar·rbar + s_t²/K_t and D_i = rbar² - s_t²/K_t - s_{t-1}²/K_{t-1}, after removing cell means.

**D5. Estimators** (one family, one common estimation sample):
- b(k) = sum(N_i - s_t,i²/k) / sum(D_i + (s_t,i² + s_{t-1},i²)/k). k = 1 is the naive limit; k to infinity is the split-sample IV. The curve in k is the noise ladder.
- `split_iv` (primary) is b(infinity), in closed form. First-stage strength is the clustered t² of mean(D). The weak-instrument interval is the Fieller set of mean(N - b0·D).
- `naive_single` uses draw 0 only; `naive_stacked` uses all single draws. `mean_of_k` averages k draws over seeded subsets for k in {1, 2, 4, 8, 16} and must track b(k).
- `cross_template_iv` (secondary, crossed block): the instrument for a template's revision is the mean revision of templates from the other five paraphrases. Same-format and other-format instruments are compared as an over-identification check. Its estimand is the coefficient on the component of beliefs common to templates. Outside the Holm families.
- `greedy`: one deterministic read, reported as the mode.
- `implied_persistence`: AKLMT's regression of the one-step forecast on the last observation, by rho. Draw noise is on the left-hand side, so no correction is needed.
- `noise_share`: draw variance over total single-draw variance.
- Per-rho estimates are ratios of summed moments within the cell. The pooled coefficient is the equal-weight mean over rho in {0.2, 0.4, 0.6, 0.8, 1}. A variance-weighted pooled regression is a robustness row.

**D6. Valid draws and units.** A draw is invalid if no number parses or if it lies more than 10 standard deviations of the shown history from the last shown value. A unit needs at least 8 valid draws at each date. The estimand is therefore the coefficient on truncated means, which the PAP states, and which leans toward echo. Robustness rows: no distance rule; winsorize at 1/99.

**D7. Templates.** 6 paraphrases x 2 number formats (comma-separated on one line; one value per line with a period index, where the prompt ends with the next index so the model continues with the value). Arm B uses the same 12 in two framings (named: variable name, forecast quarter, target quarter; anonymous: numbers only). Arm A prompts say nothing about the process. Paraphrase is the random factor; format is a fixed effect. ChronoGPT prompts are wrapped in the exact string from the authors' `extract_response`; chat models use their own chat template with no system prompt.

**D8. Template inference** is chosen by size simulation in Phase 3 from: (a) cluster by path with template fixed effects; (b) per-paraphrase estimates and a t-test with 5 degrees of freedom; (c) paraphrase-level wild cluster bootstrap. The PAP names the one with correct size and the most power; the others are robustness.

**D9. Competence gate** (proposed thresholds; AJ sets them). Gate seeds only, rho in {0, 0.5, 1}:
- parse: at least 95% of draws valid;
- history sensitivity: pairs of histories sharing the last value but generated at different rho; the difference in mean forecasts regressed on the difference in least-squares forecasts has slope at least 0.2, one-sided p < 0.01;
- persistence: implied persistence regressed on true rho has slope at least 0.25, 95% lower bound above 0;
- scale: forecasts of 3x + 500 regressed on 3F(x) + 500 have slope in [0.8, 1.2].
Before the freeze the gate must pass the least-squares learner and a history-based overreactor (implied persistence 0.45 + 0.54 x the least-squares estimate, AKLMT's pattern) and fail echo, noisy echo and constant.
Consequences, fixed now: all models are generated regardless; the gate decides inclusion. Stop rule: fewer than 3 of the 6 priority vintages pass. A failing vintage is replaced in the rolling run by the latest passing valid vintage (robustness: drop its quarters). If the 1999 or 2024 vintage fails, the test that needs it is reported descriptively. Arm B additionally requires the 95% parse rate on Arm B prompts. H3 is worded as conditional on the gate.

**D10. Hypotheses** (two-sided p-values, intervals and MDEs for all):
- H1: b(1) differs from b(infinity), pooled as in D5 over gate-passing ChronoGPT models. Report the gap, its sign, and the predicted gap from the formula.
- H2: pooled b(infinity) differs from 0; and H2-LS: it differs from the least-squares learner's on the same paths.
- H3: the slope of (implied persistence - rho) on rho, one-step forecasts, differs from 0; and H3-LS: it differs from the learner's.
- H4: pooled b(infinity) for valid-vintage forecasts differs from 0.
- H5: b under the 2024 vintage differs from b under valid vintages on the same pairs. A pair counts as contaminated when its target was first released on or before 2024-12-31.
Holm within Arm A and within Arm B. Whether the LS contrasts sit inside the Arm A family is AJ's call; default: family = {H1, H2-LS, H3-LS}, with H2 and H3 against zero reported beside them.
Benchmarks through the same code: known-parameter rational, least-squares learner (OLS of x_s on x_{s-1} with intercept, iterated for two steps), random walk, unconditional mean, each also with added noise.

**D11. Arm B construction.** Forecast quarters 2000Q2 onward while the target's first release exists. Targets follow BGMS: four-quarter growth ending t+3 for macro series, level at t+3 for unemployment and rates. Each prompt shows the last 40 quarters of the transformed series from the RTDSM vintage of its own survey quarter, one decimal. Outcome growth is computed inside the vintage where x_{t+3} first appears, unless BGMS's package shows otherwise (check in Task 2.4). Vintage rules: primary, both forecasts of a pair use the vintage valid at t-1; robustness, latest valid vintage for each. Runs: rolling valid; 1999 throughout; 2024 throughout; 2009 throughout. If not all 26 vintages complete, the pre-registered fallback is rolling on the six priority vintages. COVID (AJ confirms): primary drops pairs whose target quarter falls in 2020Q1 to 2021Q2; the full sample is a robustness row.

**D12. Arm B inference and comparisons.** Per variable: Newey-West as in BGMS. Pooled: variable fixed effects with variables scaled by their pre-2000 real-time standard deviation, weights reported, plus an equal-weight mean across variables. Candidates for pooled inference, chosen by size simulation on a moving-block bootstrap of the SPF panel under the null: Driscoll-Kraay with a fixed bandwidth and fixed-b critical values; block-t (Ibragimov-Mueller) over non-overlapping blocks. Model versus SPF: model means of K against SPF consensus, model single draws by template against SPF individuals, differences tested in one stacked system. Accuracy: RMSE against SPF and a real-time AR(4) benchmark, including the 2024-versus-valid RMSE gap. Rolling and 1999-only runs "agree" if the difference in pooled b is inside its 95% interval and below the H4 MDE. Leakage secondaries as in change 5.

**D13. Power targets.** Arm A: MDE at most 0.05 for pooled b(infinity) and for the H1 gap, at most 0.10 for the H3 slope, under the D5 estimand. Planning arithmetic: se is about 1.2/sqrt(paths per rho cell), so about 4,500 paths per cell pooled. Stop if the targets cannot be met in budget. Arm B: if the H4 MDE exceeds 0.25, the paper says the result is uninformative below the MDE and makes no "no bias" claim.

**D14. Model sets.** Arm B rolling: all 26 vintages. Arm A confirmatory: all 26 if the pilot throughput allows, else the six priority vintages (1999, 2004, 2009, 2014, 2019, 2024). The PAP states the set as a fixed list.

**D15. Ladder (descriptive).** Qwen2.5-Instruct 0.5B, 1.5B, 3B (research licence), 7B, 14B; Llama-3.2 1B and 3B (gated: AJ accepts the licence on HF). Through HF `generate` on a one-third subset of Arm A units. Capability scores: parameter count, Open LLM Leaderboard v2 average (ladder only), and zero-shot HellaSwag on a seeded 1,000-item subset for every model, run during that model's single load.

**D16. Stand-ins and pilots.** CPU pipeline: `gpt2`. Instruct-style parsing: `Qwen/Qwen2-0.5B-Instruct`. ChronoGPT architecture and throughput: `manelalab/chrono-gpt-v1-19991231` (base, not instruct). No instruct vintage is touched before the tag.

**D17. Bound to the tag.** `BOUND = ("ewh", "vendor", "prompts", "configs", "pod", "pyproject.toml", "uv.lock")`. `configs/generation.yaml` holds the master seed, K, temperature, the shard list and each model's HF revision SHA (read from the API before the tag without downloading weights). `configs/seeds.yaml` holds disjoint namespaces: pilot, dry-run and power, gate, confirmatory, subsets. `paper/` and `docs/` are outside the bound set.

---

## Repository

`~/Documents/expectations-without-hindsight` (does not exist; `git init`, then `change_directory`). Handoff Section 7 layout plus:

```
vendor/chronogpt/          ChronoGPT_instruct.py from the pinned HF revision, LICENSE, sha256
prompts/                   templates.yaml and built prompt tables (parquet)
configs/                   generation.yaml, seeds.yaml
ewh/provenance/            archive.py (sidecars), manifest.py (sha256 walk and verify)
ewh/confirm/binding.py     tag, clean tree, blob and input-hash checks; attempt marker; write-once
ewh/models/chrono_fast.py  batched cached sampler for ChronoGPT
ewh/models/hf_backend.py   transformers backend for gpt2 and the ladder
ewh/models/ledger.py       append-only generation ledger
ewh/capability/            competence gate, HellaSwag scorer
pod/                       setup.sh, download.sh, run_session.sh (stop-pod trap), sync.sh
paper/make_inputs.py       numbers.tex and tables from result files
docs/pilot_log.md, gpu_spend_log.csv
```

### Code to reuse (read first; copy and adapt with a provenance comment)

| Need | Source |
|---|---|
| Tag-bound runner, attempt marker, write-once, tests | `~/Documents/sharp-markets/src/sharpmkt/confirm.py` (`check_binding` L69, `create_marker` L102, `write_once` L113), `tests/test_confirm.py` |
| Input hashing | `~/Documents/haircut/haircut/manifest.py` (`build_manifest` L44, `canonical_json` L76), `haircut/panel/content_hash.py` (`frame_hash` L62), `haircut/panel/seal_log.py` (hash-chained log) |
| PAP structure | `~/Documents/sharp-markets/docs/pap/pre-analysis-plan.md` |
| Decision and assumption logs | `~/Documents/gpu-econ-life/docs/decisions.md`, `assumptions.md` |
| Raw archive, sha256 manifest | `~/Documents/gpu-econ-life/src/gel/collect/http.py` (`write_raw` L74), `src/gel/release/manifest.py` (`build` L75, `verify` L106, `deposit` L123) |
| Two-way clustering, wild cluster bootstrap | `~/Documents/gpu-econ-life/src/gel/models/hedonic_core.py` (`twoway_se` L128, `wild_cluster_bootstrap_restricted` L221) |
| Newey-West | `~/Documents/haircut/haircut/engine/metrics.py` (`newey_west_cov` L30) |
| Coverage simulation on the real design | `~/Documents/gpu-econ-life/src/gel/models/review_checks.py` (`monte_carlo` L242) |
| Generated paper numbers | `~/Documents/gpu-econ-life/src/gel/viz/paper_numbers.py` (`Numbers` L24); `~/Documents/sharp-markets/paper/make_inputs.py` |
| Resume by part file | `~/Documents/pm-diffusion/src/pmd/data/midpoints.py` (`run` L109, `merge_parts` L129) |
| Pod hygiene, spend log | `~/Documents/gpu-econ-life/docs/bench_runbook.md`, `bench/run_all.sh` |
| Adversarial review format | `~/Documents/gpu-econ-life/docs/review_01.md`, `docs/proposal.md` L436 |
| Web entry pattern | esnupi `origin/main` commit `56c322c` |

Written new: Driscoll-Kraay with fixed-b critical values, block-t, Holm, Fieller intervals, the ChronoGPT sampler, the estimator family.

### Core interfaces

```python
# ewh/estimators/types.py
@dataclass(frozen=True)
class Draws:
    f_t: np.ndarray       # (U, K) forecasts at t of the target; NaN = invalid draw
    f_tm1: np.ndarray     # (U, K) forecasts at t-1 of the same target
    x: np.ndarray         # (U,) realized target
    cell: np.ndarray      # (U,) cell id (rho, or variable)
    clusters: np.ndarray  # (U, C); column names in cluster_names
    cluster_names: tuple[str, ...]

@dataclass(frozen=True)
class Estimate:
    b: float; se: float; ci: tuple[float, float]; p: float
    n_units: int; n_clusters: tuple[int, ...]
    first_stage_t2: float | None; fieller: tuple[float, float] | None
    psi: np.ndarray       # (U,) unit-level influence values

# ewh/estimators/eor.py
def unit_moments(d: Draws, min_valid: int = 8) -> UnitMoments        # ebar, rbar, s2_t, s2_tm1, K_t, K_tm1, N, D
def b_of_k(m: UnitMoments, k: float, cov: CovSpec) -> Estimate       # k = math.inf is split_iv
def split_iv(d: Draws, cov: CovSpec) -> Estimate
def split_iv_enumerated(d: Draws) -> float                           # test-only: all splits
def naive_single(d: Draws, cov: CovSpec, draw: int = 0) -> Estimate
def naive_stacked(d: Draws, cov: CovSpec) -> Estimate
def mean_of_k(d: Draws, k: int, cov: CovSpec, n_subsets: int = 50, seed: int = 0) -> Estimate
def cross_template_iv(d: Draws, paraphrase: np.ndarray, fmt: np.ndarray, path: np.ndarray, cov: CovSpec) -> Estimate
def pooled_equal_weight(by_cell: dict[float, Estimate], cells: tuple[float, ...]) -> Estimate
def difference(a: Estimate, b: Estimate, clusters: np.ndarray, cov: CovSpec) -> Estimate
# ewh/estimators/persistence.py
def implied_persistence(f_mean: np.ndarray, x_last: np.ndarray, cov: CovSpec) -> Estimate
def noise_share(d: Draws) -> float
# ewh/inference/
def cov_from_psi(psi, clusters, spec: CovSpec) -> float   # CR1, wild bootstrap, Driscoll-Kraay fixed-b, Newey-West, block-t
def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]
def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float
# ewh/sim/
def ar1_paths(rho: float, n_paths: int, length: int, seed: int, sd: float = 20.0) -> np.ndarray
def rational(hist, rho, h); def ls_learner(hist, h, intercept=True); def random_walk(hist, h); def uncond_mean(hist, h)
def planted(hist, h, theta: float, noise_sd: float, k: int, rng) -> np.ndarray
# ewh/prompts/
def build_arm_a(seeds, templates) -> pd.DataFrame; def build_gate(seeds, templates) -> pd.DataFrame
def build_arm_b(panel: RealTimePanel, templates) -> pd.DataFrame
def parse_number(text: str, fmt: str) -> float | None
# ewh/models/
class Backend(Protocol):
    def sample(self, prompt_ids: list[str], prompts: list[str], k: int, master_seed: int,
               temperature: float, max_new_tokens: int) -> SampleBatch: ...   # texts, first-token top-64 logprobs per prompt
    def greedy(self, prompts: list[str], max_new_tokens: int) -> list[str]: ...
def run_shards(backend: Backend, prompt_table: Path, out_dir: Path, cfg: GenerationConfig, ledger: Ledger) -> None
# ewh/confirm/binding.py
def check_binding(tag: str = "pap-v1") -> datetime
def check_inputs(manifest: Path, generation_cfg: Path) -> None   # exactly the pre-registered shard set
def create_marker(family: str) -> None
```

Tables: draws (`prompt_id`, `arm`, `model_repo`, `model_revision`, `code_commit`, `master_seed`, `draw` with -1 for greedy, `temperature`, `raw_text`, `parsed`, `valid`, `shard`); prompts-run (`prompt_id`, `model_revision`, `n_prompt_tokens`, `first_token_logprobs` top 64, `gpu`, `driver`, `torch_version`, `created_utc`).

---

## Phases and tasks

Each task: tests first, `uv run pytest` and `uv run ruff check` green, commit.

### Phase 0: gate, reading, access

**0.1 Scaffold.** `git init`, noreply author, uv project, `Makefile` (`test`, `lint`, `all`), `CLAUDE.md` (gpu-econ-life's rules adapted), `docs/HANDOFF.md`, `docs/PLAN.md`, empty `decisions.md`, `assumptions.md`, `deviations.md`, `pilot_log.md`. Create the private GitHub repo after AJ says so.
**0.2 Novelty check** in `docs/novelty_check.md`: every hit, date, overlap with C1 to C4. Start from the pre-scan in Context; cover SSRN and Google Scholar in a browser. **Stop if C1 and C2 are both published.**
**0.3 Read and verify the core papers** into `docs/references_verified.md` (title, authors, year, venue, DOI, the claim relied on, page). Archive the AKLMT and BGMS packages in `data/raw/`; extract AKLMT's error-on-revision coefficients by rho from their data.
**0.4 Data access and terms** in `docs/data_terms.md`; fetch one SPF and one RTDSM file through `ewh/provenance/archive.py`. **Stop if any source is not publicly usable.**
Tests: `test_archive_writes_sidecar_and_refuses_overwrite`, `test_manifest_verify_detects_changed_byte`.

### Phase 1: simulator, estimators, the noise result

**1.1 Simulator and benchmarks.** Tests: sample autocorrelation matches rho; `rational` equals rho^h x last; `ls_learner` matches a hand regression; separate rho streams share no shocks; seeds reproduce.
**1.2 Estimator family.** Tests on planted forecasters (bias theta, noise sd by date):
- `test_naive_matches_general_formula`, with unequal s_t and s_{t-1};
- `test_attraction_point`: at b* = -s_t²/(s_t² + s_{t-1}²) naive equals b*; above it naive is lower, below it higher;
- `test_split_iv_equals_enumeration`: closed form equals the average over all splits at K = 4 to 1e-10, with cell demeaning;
- `test_split_iv_coverage`: 95% intervals cover the planted coefficient in 93 to 97% of 2,000 replications, for theta in {-0.7, -0.3, 0, 0.3} and noise share in {0, 0.3, 0.6}. **The handoff's required recovery test.**
- `test_mean_of_k_tracks_b_of_k`;
- `test_zero_draw_variance_is_finite`;
- `test_weak_first_stage_flagged`: rational at rho = 0.2 with heavy noise gives a small t² and a wide or unbounded Fieller set;
- `test_ls_learner_is_not_zero`: the learner's b at rho = 0.2 is near -0.35 and its H3 slope is negative, so the LS contrasts are needed;
- `test_common_sample`: every estimator drops the same units.
**1.3 Inference.** Driscoll-Kraay and Newey-West match statsmodels on a fixed panel; wild bootstrap matches gpu-econ-life's fixture; Holm matches a worked example; `difference` reproduces a paired test.
**1.4 Derivation and Monte Carlo.** `docs/noise_derivation.md`: the general limit, the attraction point, consistency of split-sample IV, the closed form, the learner's nonzero b. `make noise-mc` draws the noise ladder for planted forecasters.
**Stop if split-sample IV fails coverage.**

### Phase 2: data and the human benchmark

**2.0 Commit the Arm B choices** (D11, D12 as text in `docs/decisions.md`) before any SPF regression runs.
**2.1 Loaders** for SPF (consensus, individual, deadlines), RTDSM vintage matrices, FRED rates; all archived with sidecars.
**2.2 Real-time sets.** Tests: `test_history_from_single_vintage`; `test_outcome_growth_within_one_vintage`; `test_no_ratio_spans_rebasing` (2003, 2009, 2013, 2018, 2023); one hand-checked RGDP quarter.
**2.3 Pair builder.** Tests: `test_pair_never_mixes_vintages`; `test_first_pair_is_2000Q2`; `test_q1_uses_lagged_vintage`; `test_failed_vintage_substitution`; `test_six_vintage_fallback_assignment`.
**2.4 SPF replication** on BGMS's sample, compared with their package output. Tolerance fixed before running: for NGDP, RGDP, PGDP, CPI, UNEMP, TB3M and TN10Y, each consensus coefficient within 0.15 of published with the same sign for at least 6 of 7; CG inflation in [1.0, 1.5]. `docs/spf_replication.md`. **Stop if this fails.**

### Phase 3: prompts, sampler, dry run, power, pilot

**3.1 Templates and prompt tables.** Write and hash `prompts/templates.yaml` first. Build Arm A (main and crossed), gate and Arm B tables. `docs/tokenization.md`: for every tokenizer, how each integer in range, negatives and one-decimal values split; longest prompt against 1,792. Read each ladder model's `generation_config.json` and record what must be overridden. Tests: tables rebuild byte-identical; none exceeds context; Arm B prompts contain no outcome; seed namespaces are disjoint.
**3.2 Parser and validity.** Fixture of awkward outputs: words before the number, two numbers, a trailing unit, a split negative sign, empty text, a leading newline, a wild number, and the period-index echo ("41. 23" must give 23 in the index format).
**3.3 ChronoGPT sampler.** Vendor the authors' file unedited. Write a separate forward with a dtype argument, per-layer key/value cache, last-position logits, right padding, and K continuations that attend to the prompt cache by view. Points that diverge silently, each with a test: rotary positions need an absolute offset for incremental tokens; `is_causal=True` is wrong for a single query against a cache; value embeddings are indexed by token id and enter v before caching; half the rotary dimensions have zero frequency; the logit cap 15·tanh(x/15); fp16 versus bf16 parameter dtypes. Tests on a tiny config **with every parameter randomised** (construction zero-initialises the output layers, which would make every test pass vacuously):
- `test_fp32_cached_equals_fp32_uncached` to 1e-4 at every step, over several steps with unequal prompt lengths in one batch;
- `test_bf16_matches_vendored`: KL below a fixed bound and top-1 agreement except where the top-two margin is below a fixed bound;
- `test_k_draws_independent`: draw k's continuation never conditions on draw j's prefix;
- `test_batch_invariance`;
- `test_rng_streams`: identical prompts give different draws; t and t-1 draws are uncorrelated; reruns reproduce;
- `test_sampling_matches_softmax` (chi-square).
On this Mac: time one 64-token forward of the base model first; run at most a few short prompts. Full-weight equivalence is done in the pilot. There is no slow fallback: if equivalence fails, stop and report.
**3.4 Shard runner, ledger, HF backend.** Tests: kill mid-shard and resume equals an uninterrupted run; a failed shard is logged and redone under the same seed, never replaced with a new one; the effective HF generation config equals D3 (no repetition penalty, no top-k); schema validates.
**3.5 Gate and capability code.** Tests: the gate passes the least-squares learner and the history-based overreactor and fails echo, noisy echo and constant.
**3.6 CPU dry run** on dry-run seeds with `gpt2` and planted forecasters behind `Backend`: prompts, generation, ledger, parsing, gate, estimators, both confirm scripts in `--dry-run`, tables, figures, PDF. Planted bias comes back under every estimator.
**3.7 Size and power** (`make power`, power seeds): choose D8 and D12 methods by simulated size; MDEs for every hypothesis on a grid of paths per unit and K, with noise calibrated to the non-confirmatory pilot models; the allocation table. `docs/power.md`. **Stop if the Arm A MDE is not small or the budget cannot cover the confirmatory set.**
**3.8 GPU pilot** (approved; about 30 minutes, under $2): base model, pilot seeds, rho = 0.5. Full-weight equivalence against the authors' `generate` (two-sample test of parsed numbers on 200 prompts), tokens per second at the planned batch sizes, model load time from the volume. Results to `docs/pilot_log.md` and `docs/compute_budget.md`. The PAP's K, paths and vintage sets are then fixed numbers.

### Phase 4: pre-analysis plan and freeze

**4.1 `docs/PAP.md`** on the sharp-markets outline. Decisions for AJ at the top, each with a recommendation and its alternative: changes 1 to 6 above, gate thresholds, named versus anonymous as primary, variable list, COVID rule, Holm family membership, title. Then disclosures (pilots, benchmark runs, SPF replication, the planning pre-scan), outcomes, samples, estimators, inference, templates with hashes, seeds, robustness rows, the allocation, allowed in-session diagnostics (row counts, parse rate, tokens per second, sampler preflight), the bound-blob table, deviation rules.
**4.2 Confirm scripts** `gate.py`, `arm_a.py`, `arm_b.py`. Tests extended from sharp-markets: lightweight tag refused; dirty tree refused; changed input refused; missing or extra shard refused; second run refused; dry run cannot read confirmatory files.
**4.3 AJ signs off; tag `pap-v1`; push; ask AJ, then make the repo public.** Always a stop.

### Phase 5: the GPU session (runbook below)

### Phase 6: estimation, robustness, review

**6.0** Commit and push `data/generations/MANIFEST.json` and the ledger before any estimate. **6.1** `gate.py` (must reproduce the hashed gate file written on the pod), then `arm_a.py` and `arm_b.py`, once each. **6.2** Robustness and descriptive rows: ladder, framing, by variable, horizons, constant-capability check, leakage secondaries, RMSE against SPF and the AR benchmark, model-minus-SPF. **6.3** Adversarial review: a fresh session given only `ewh/`, the result tables and the PAP's identification section writes `docs/review_01.md`; each point answered in `docs/review_01_checks.md` with `make review-checks`. Changes after seeing results go in `docs/deviations.md` and demote the hypothesis.

### Phase 7: paper, package, web

**7.1** LaTeX paper from generated `numbers.tex` and tables; `[AJ]` marks with a public-build flag; limitations lead with what the design cannot establish; AI-use statement; data and code availability; licences (including "Built with Qwen" if required). **7.2** Clean-clone `make all` with the network off. **7.3** Reference check. **7.4** Deposit raw generations (Zenodo, as gpu-econ-life's `deposit`), after AJ approves. **7.5** Web summary: branch esnupi from `origin/main` in a worktree and follow commit `56c322c`; the local checkout is 21 commits behind with 44 uncommitted changes for a different publication system, so ask AJ which is current. Verify in a browser.

---

## GPU session runbook (Phase 5)

**Before:** AJ confirms the balance and gives a go on the quoted price. Install `runpodctl` or use the RunPod MCP tools for lifecycle and ssh/rsync for work. Re-read live stock and price.

**Hardware:** an 80 GB card in a data center with network volumes (A100 80GB at $1.79/h today; H100 if one appears there).

**Weights:** 220 GB network volume (about $3.60 for a week); download from a CPU pod in priority order with `HF_HUB_DISABLE_XET=1` and retries; verify sha256 and the revision SHAs in `configs/generation.yaml`. If no volume-capable data center has a suitable GPU, download to the GPU pod's disk instead.

**Order** (repo cloned at `pap-v1`; `run_session.sh` checks the tag and clean tree, writes the ledger, traps exit to stop the pod):
1. Throughput check, 10 minutes; confirm the PAP's allocation in `docs/compute_budget.md`.
2. Six priority vintages, one load each: sampler preflight, gate, Arm A, HellaSwag, and the Arm B rows of the six-vintage fallback rolling run. The 1999, 2024 and 2009 vintages also run every Arm B quarter. Compute the gate; **stop the pod and report if fewer than 3 of 6 pass.**
3. The other 20 vintages, one load each: preflight, gate, Arm A (if in the set), HellaSwag, their annual rolling Arm B rows.
4. Ladder models, smallest first. Cut first if time runs short.
Rsync parquet and the ledger to this Mac after every shard. At the end: verify row counts and hashes locally, stop the pod, confirm with `list-pods`, log spend. Delete the volume only after AJ confirms.

**Provisional sizing** (replaced by Task 3.7): Arm A main block about 7,200 prompts per vintage (600 paths per rho); crossed block about 14,400 prompts for each of two vintages; gate about 3,000 per model; Arm B about 69,000 prompts per full run, five runs (annual rolling, six-vintage rolling, and the 1999, 2009 and 2024 vintages throughout). About 0.65 million prompts in all, each with 16 draws and a greedy read. At a cached throughput of 30 to 60 prompts per second on an A100 that is 3 to 6 hours plus about half an hour of model loads.

**Spend estimate:** pilot $2, storage $4, CPU pod $1, session up to 8 A100-hours $14: about $21, inside $50. An H100 is faster at a similar total.

---

## Verification

- `uv run pytest` green at every commit; clean-clone `make all` with the network off.
- Phase 1: coverage, enumeration and attraction-point tests pass; the planted-forecaster noise ladder matches the derivation.
- Phase 2: `docs/spf_replication.md` shows our coefficients beside BGMS's and CG's inside tolerance.
- Phase 3: sampler tests pass on the randomised tiny config; the pilot shows full-weight equivalence; the CPU dry run yields a PDF whose numbers match what was planted.
- Phase 4: the confirm tests prove refusal of a lightweight tag, a dirty tree, a changed input, a wrong shard set and a second run.
- Phase 5: local rows equal prompts x 17 per model; every `model_revision` matches `configs/generation.yaml`; the ledger has no replaced shard; nothing is running on RunPod.
- Phase 6: every hypothesis has an estimate, interval, two-sided p-value and MDE whether or not it rejects.
- Phase 7: a script fails the build on any digit in the paper source outside generated macros; every reference has a verified entry.

## Sessions and stop points

| Session | Work | Ends with |
|---|---|---|
| 1 | Phase 0 | Novelty verdict and data access; stop if blocked |
| 2 | Phase 1 | Estimator family proven on planted bias |
| 3 | Phase 2 | SPF replication; stop if it fails |
| 4 to 5 | Phase 3 | Dry run, power, GPU pilot; stop if MDE or budget fails |
| 6 | Phase 4 | PAP for AJ's sign-off, then the tag |
| 7 | Phase 5 | Generations on disk, pod stopped |
| 8 | Phase 6 | Results, robustness, review answered |
| 9 | Phase 7 | PDF, package, web summary |

## Decisions AJ owns

Settled on 2026-11-11:
1. **$50 binds**, storage included; the GPU is chosen by live stock and price.
2. **The pre-freeze GPU pilot is approved** (Task 3.8).
3. **The repo is private while building and public when `pap-v1` is pushed.** Ask again at that moment.

For the PAP (Phase 4): changes 1 to 6 above (the benchmark for "bias", two-sided H1, the equal-weight pooled estimand without rho = 0, nested paths, the leakage secondaries, the COVID rule); final wording of H1 to H5; gate thresholds; named versus anonymous as primary (default named); the variable list; whether to add credit; the title.
