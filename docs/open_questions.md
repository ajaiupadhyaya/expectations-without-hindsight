# Open questions

Values that are unknown, unreachable or unverified. Each is resolved or carried into the PAP.

## OQ1 (2026-11-11): BGMS and CG replication packages
openICPSR (BGMS doi:10.3886/E118422V1; CG doi:10.3886/E112862V1) blocks automated access
(bot check, HTTP 403) and requires a free sign-in to download. Claude does not create or use
accounts. **Ask AJ** to download both zips with their own account and place them in
`~/Downloads/`; they are then archived with a sidecar marking manual retrieval.
Needed for: Phase 2.4 replication check against package output, and D11's question of how
BGMS compute growth outcomes across NIPA rebasing. Not blocking Phase 0 or 1: the published
Table 3 values (Aug 2019 revision) and the AKLMT package's own processed SPF panel
(`data/SPF/spf_out_trim5.dta`, BGMS-style, built by co-author Ma) are available meanwhile.

## OQ2 (2026-11-11): Arm B prompts when the survey-quarter vintage is stale — for the PAP
Two survey quarters have vintages that end one quarter early because releases were delayed by
government shutdowns: 2025Q4 for every NIPA series and housing starts, and 2013Q4 for housing
starts. A prompt built from the vintage alone would show a history ending at s−2. The SPF itself
notes the 2025Q4 survey lacked jump-off values. Options: (a) drop the affected pairs (1 of 103
quarters for NIPA series, 2 of 102 for housing); (b) keep them with the shorter history and a
6-step (t) / 5-step (t−1) horizon. **Proposed default: (a)**, decided in the PAP before any model
output exists. `realtime.history` raises on a stale vintage, so nothing is silently filled.
