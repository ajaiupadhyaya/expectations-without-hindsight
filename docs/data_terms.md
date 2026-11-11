# Data sources and terms of use

Checked 2026-11-11. Every source below is public and was downloaded without a login by
`ewh.data.sources` (sidecars under `data/raw/`). **Verdict: all data the confirmatory analysis
needs is publicly usable for research. No Phase 0 stop.**

## Philadelphia Fed: Survey of Professional Forecasters and Real-Time Data Set

| Item | Detail |
|---|---|
| Files tested | `Individual_RGDP.xlsx` (9,280 rows, 1968Q4–2026Q3); `spf-release-dates.txt` (survey deadlines 1990Q2–2026Q4, with notes on delayed surveys and the 2025Q4 shutdown); `spf-documentation.pdf`; `ROUTPUTQvQd.xlsx` (vintages 1965Q4–2026Q4, observations 1947Q1–2026Q3). All parse. |
| Access | Direct `/-/media/` links, no login. Each link carries a `hash=` parameter that changes when the file is updated, so loaders read the current link from the landing page and snapshots are stored under their retrieval date. |
| robots.txt | Disallows only `/404`, `/502`, `/api/`, `/error`, `/search-results`. Data files are not disallowed. |
| Terms | The site's "Appropriate Use" section (on `/about-us/privacy-notice`) allows use of site content "for informational, educational, and research purposes only", forbids excessive access that could disrupt the site, and forbids use of the Bank's name as an endorsement. No redistribution clause. |
| Citation | RTDSM: Croushore and Stark (2001), *Journal of Econometrics* 105, 111–130, which the RTDSM page names as the preferred citation. SPF: cite the Philadelphia Fed SPF page and documentation; no specific article requested. |
| Implication | Research use is allowed. The replication package archives raw files with checksums and points to the source URLs. Redistributing the raw files in the deposit is not addressed by the terms; to be conservative, the Phase 7 deposit includes the files with attribution only if AJ agrees, otherwise it ships the checksums and a re-download script. (Decision for Phase 7.) |
| Technical note | `ROUTPUTQvQd.xlsx` has a malformed document-properties date that `openpyxl` rejects. Loaders read Philadelphia Fed workbooks with the `calamine` engine (`python-calamine`). |
| Data note | Real GDP levels differ across vintages because of NIPA rebasing (1947Q1 is 306.4 in the 1965Q4 vintage and 2,182.7 in the 2026Q4 vintage). This confirms that growth outcomes must be computed within one vintage (plan D11, Task 2.2). |

## FRED (Federal Reserve Bank of St. Louis)

| Series | Source and status | Use |
|---|---|---|
| TB3MS (3-month T-bill) | Board of Governors, H.15; FRED marks it **public domain** | Arm B rate variable (never revised, so no vintages needed) |
| GS10 (10-year Treasury, constant maturity) | Board of Governors, H.15; FRED marks it **public domain** | Arm B rate variable |
| AAA (Moody's Aaa yield) | Moody's; FRED marks it **copyright, citation required**; FRED's legal page says use beyond personal use needs the owner's permission | **Excluded** (plan change 14). The SPF public files also omit Moody's history. |

FRED CSV downloads (`fredgraph.csv?id=…`) need no API key. FRED asks users to name FRED as the
source and keep any copyright notices; cite each series with FRED's suggested citation.

## Replication packages

| Package | Licence / access | Status |
|---|---|---|
| AKLMT (QJE 2023), Harvard Dataverse doi:10.7910/DVN/3Q6BHC | CC0 1.0, open | Archived; publisher md5 verified |
| BGMS (AER 2020), openICPSR doi:10.3886/E118422V1 | AEA data policy; openICPSR requires a free sign-in; site blocks automated access | Not obtained (open question OQ1) |
| CG (AER 2015), openICPSR doi:10.3886/E112862V1 | as above | Not obtained (OQ1) |

## Models (for Phase 3 onward; recorded now so licences are known before the plan is frozen)

| Model | Licence | Note |
|---|---|---|
| `manelalab/chrono-gpt-instruct-v1-YYYY1231` (26 vintages), `chrono-gpt-v1-19991231` (pilot) | MIT | Open, not gated |
| Qwen2.5-Instruct 0.5B, 1.5B, 7B, 14B | Apache-2.0 | Open |
| Qwen2.5-3B-Instruct | Qwen Research License (non-commercial, research and evaluation) | Academic use allowed; "Built with Qwen" notice if a model trained on its outputs is distributed (we distribute none) |
| Llama-3.2 1B and 3B Instruct | Llama 3.2 Community License | Gated: AJ must accept the licence on Hugging Face before Phase 5 |
| `gpt2`, `Qwen/Qwen2-0.5B-Instruct` (stand-ins) | MIT / Apache-2.0 | CPU dry run and parsing pilots only |
