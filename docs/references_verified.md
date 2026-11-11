# References verified

Every reference the paper relies on, checked against an authoritative record before use.
Metadata were checked on 2026-11-11 against Crossref (journal DOIs) or the arXiv API (preprints);
both record sources are machine-readable and the queries are reproducible. "Claim relied on" is
what the paper will say about the work. "Where" is the location of that claim. Entries marked
**page to confirm** were read in a working-paper or abstract version; the exact page in the
published version is checked before the draft (Phase 7.3).

`paper/references.bib` is generated from this file's metadata and must stay in sync.

## Core methods and human benchmarks

| Key | Verified metadata | Claim relied on | Where |
|---|---|---|---|
| CG2015 | Coibion, O. and Gorodnichenko, Y. "Information Rigidity and the Expectations Formation Process: A Simple Framework and New Facts." *AER* 105(8), 2644–2678, 2015. doi:10.1257/aer.20110306 | Consensus error-on-revision regression; SPF inflation coefficient about 1.2 (underreaction); actuals are real-time data one year later; Newey–West errors. | Spec in Section I; coefficient in NBER WP 16537 (1969–2010) is 1.23 (0.50). **Published table value to confirm.** |
| BGMS2020 | Bordalo, P., Gennaioli, N., Ma, Y. and Shleifer, A. "Overreaction in Macroeconomic Expectations." *AER* 110(9), 2748–2782, 2020. doi:10.1257/aer.20181219. Earlier: NBER WP 24932, 2018 (doi:10.3386/w24932) | 22 series; horizon t+3; growth t−1 to t+3 for macro series, level for rates; outcomes from the real-time release in t+4; consensus errors Newey–West, individual errors two-way clustered by forecaster and time; individual coefficients mostly negative, consensus mostly positive. | Section I and Table 3. Values read from the Aug 2019 revision, whose individual-FE column equals published appendix Table C2. **Published Table 3 page to confirm.** |
| AKLMT2023 | Afrouzi, H., Kwon, S. Y., Landier, A., Ma, Y. and Thesmar, D. "Overreaction in Expectations: Evidence and Theory." *QJE* 138(3), 1713–1764, 2023. doi:10.1093/qje/qjad009. Replication data: Harvard Dataverse doi:10.7910/DVN/3Q6BHC (CC0 1.0) | AR(1) design: mean 0, sd 20, rho in {0, .2, .4, .6, .8, 1}, 40 past values, forecasts of t+1 and t+2; implied persistence 0.45 to 0.99 (Exp. 1, h=1); error-on-revision negative, more so at low rho; noise in F "could" make b mechanically negative. | Design Section II; implied persistence Figure II B (**reproduced exactly from their data**, `ewh/data/aklmt.py`); noise remark §2.2 of the Feb 2023 working paper (**page to confirm**). |
| KP2024 | Kučinskas, S. and Peters, F. S. "Measuring Under- and Overreaction in Expectation Formation." *REStat* 106(6), 1620–1637, 2024. doi:10.1162/rest_a_01255. (Earlier title "Measuring Biases in Expectation Formation", Tinbergen DP TI 2018-058/IV.) | IV estimator for under/overreaction consistent under measurement error in forecasts. | **Page to confirm.** |
| JK2023 | Juodis, A. and Kučinskas, S. "Quantifying Noise in Survey Expectations." *Quantitative Economics* 14(2), 609–650, 2023. doi:10.3982/QE1633 | Noise is 10–30% of SPF forecast MSE. | Abstract; **page to confirm.** |
| CS2001 | Croushore, D. and Stark, T. "A Real-Time Data Set for Macroeconomists." *Journal of Econometrics* 105(1), 111–130, 2001. doi:10.1016/S0304-4076(01)00072-0 | Source and preferred citation for the RTDSM. | Philadelphia Fed RTDSM page. |

## Vintage models

| Key | Verified metadata | Claim relied on | Where |
|---|---|---|---|
| HLMW2026 | He, S., Lv, L., Manela, A. and Wu, J. "Chronologically Consistent Large Language Models." *Journal of Financial Economics* 185, 104374, 2026. doi:10.1016/j.jfineco.2026.104374. Preprint arXiv:2502.21206 (v3, 2025-07-06); SSRN 5159615 | ChronoGPT/ChronoBERT base models with annual cutoffs 1999–2024; pretraining data restricted by date; GLUE and HellaSwag comparisons. | **Page to confirm** (cite the JFE version; it appeared after the handoff was written). |
| HLMW2025 | He, S., Lv, L., Manela, A. and Wu, J. "Instruction Tuning Chronologically Consistent Language Models." arXiv:2510.11677 (v2, 2025-11-17). SSRN 5348747 ("Chronologically Consistent Generative AI") | Instruct models; the same instruction set for every vintage, screened to pre-2000 by a GPT-4.1 classifier; prompt template; vintages do not recall post-cutoff presidents or events. | §2.2.1, §3.3 of v2. |
| DatedGPT | Yan, Y., Tang, R., Gao, Z., Jiang, W. and Lu, Y. "DatedGPT: Preventing Lookahead Bias in Large Language Models with Time-Aware Pretraining." arXiv:2603.11838 (v2, 2026-07-23) | Alternative vintage family (1.3B, cutoffs 2013–2024). | Abstract. |

## LLM expectations and lookahead

| Key | Verified metadata | Claim relied on | Where |
|---|---|---|---|
| Bybee2023 | Bybee, L. "Surveying Generative AI's Economic Expectations." arXiv:2305.02823 (v2, 2023-05-05); SSRN 4430515 | GPT-3.5 expectations vs SPF with CG regressions; commercial model with an in-sample cutoff. | Abstract and Section 4. |
| CGGZ2024 | Chen, S., Green, T. C., Gulen, H. and Zhou, D. "What Does ChatGPT Make of Historical Stock Returns? Extrapolation and Miscalibration in LLM Stock Return Forecasts." arXiv:2409.11540 (v1, 2024-09-17); Dec 2025 draft on the AEA 2026 program | LLM over-extrapolation stronger for less persistent series. | Abstract. (The handoff's "2025" refers to the AEA draft; cite the arXiv version and year 2024 unless a journal version appears.) |
| GJY2026a | Gao, Z., Jiang, W. and Yan, Y. "Debiasing LLMs by Fine-tuning." arXiv:2604.02921 (v2, 2026-05-02) | AKLMT design on Qwen3-32B at temperature 0, one forecast per prompt; b from −0.456 (rho=0) to −0.260 (rho=1); fine-tuning removes it. | HTML v2, Table on Experiment 1. |
| GJY2026b | Gao, Z., Jiang, W. and Yan, Y. "Detecting Lookahead Bias in LLM Forecasts." arXiv:2512.23847 (v2, 2026-06-12; v1 2025-12-29 titled "A Test of Lookahead Bias in LLM Forecasts") | Test for lookahead bias in LLM forecasts. | Abstract. Cite v2's title. |
| ES2026 | Eliseev, A. and Seleznev, S. "Fake Date Tests: Can We Trust In-sample Accuracy of LLMs in Macroeconomic Forecasting?" arXiv:2601.07992 (v2, 2026-03-30); SSRN 6035916 | Fake-date tests show in-sample macro accuracy is contaminated. | Abstract. |
| BCHJ2026 | Bini, P., Cong, L. W., Huang, X. and Jin, L. J. "Behavioral Economics of AI: LLM Biases and Corrections." NBER WP 34745, 2026 (doi:10.3386/w34745); arXiv:2602.09362 | AKLMT experiments on commercial LLMs; implied persistence; large models near rational at h=1. | Abstract and Section on expectations. |
| JA2026 | Jaworski, K. and Alam, M. J. "Synthetic Consensus: Evaluating Large Language Model Forecasts with Repeated Draws." SSRN 7382959, dated 2026-08-01 | Averaging repeated draws cuts LLM inflation-forecast error by roughly 7–35%; more capable models vary less. | Abstract only (SSRN blocks automated access). **Read full text before citing numbers.** |
| GHJ2026 | Graham, J. R., Harvey, C. R. and Jha, M. "CFOs Meet LLMs." arXiv:2606.13812 (v2; v1 2026-06-11); SSRN 6920881 | Run-to-run variation is small relative to between-prompt variation; averaging. | Abstract and pre-scan reading of the PDF; **page to confirm.** |
| CE2026 | Canen, N. and Enamorado, T. "When Predictions Become Regressors: A Split-Sample Correction for Biases in Downstream Inference." arXiv:2608.02909 (v1, 2026-08-03) | Split-sample IV for predicted regressors; methodological cousin of our estimator. | Abstract. |

## Not yet obtained

- BGMS and CG replication packages (openICPSR E118422V1 and E112862V1): the openICPSR site blocks
  automated access and requires a sign-in to download. See `docs/open_questions.md`.
