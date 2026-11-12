# Compute budget (DRAFT until the GPU pilot; finalized before tag pap-v1)

Hard cap: $50 of RunPod credit in total, storage included (AJ, 2026-11-11).

## Spent so far
| Date | Item | Cost |
|---|---|---|
| 2026-11-12 | GPU pilot, RTX 4090, pod r5zmy0q27169tc, 01:27–02:31 UTC (deleted, confirmed by list-pods) | about $0.95 (billing not yet posted) |

## GPU pilot (Task 3.8) — done 2026-11-12 (docs/pilot_log.md, Pilot 2)
- Model: `manelalab/chrono-gpt-v1-19991231` (non-instruct base, non-confirmatory), pilot seeds,
  rho = 0.5, 480 prompts. Measures: sampler equivalence on full weights, parse rate,
  prompts per second by batch size and K, load time. No forecast statistics.
- Proposed GPU: RTX 4090 24 GB, Secure Cloud, EU-RO-1 ($0.89/hr from the 2026-11-11 catalog
  read; stock HIGH overall, MEDIUM in EU-RO-1). Expected 30–45 min including the 7.4 GB
  download: about $0.50–0.70. No network volume.
- Code reaches the pod by rsync over SSH (pod-level PUBLIC_KEY variable; no account change).

## Confirmatory session — sizes to be fixed from the pilot throughput
| Block | Prompts | Notes |
|---|---|---|
| Arm A main, per vintage | 24 templates x 6 rho x n paths x 2 dates | n from docs/power.md |
| Arm A crossed block | 2 vintages x 24 templates x 6 rho x n_c x 2 | for the cross-template IV |
| Gate, per model | 7 kinds x n_g | |
| Arm B, per full run | 137,088 (templates v2) | 5 runs (annual rolling, six-vintage rolling, 1999, 2009, 2024 throughout) |
| Ladder | one third of Arm A units | descriptive; cut first |

## What the pilot implies for the confirmatory session (RTX 4090 figures; an H100 is faster)
At K = 32 and batch 32, about 30 prompts/s on Arm A prompts (Arm B prompts are ~40% longer;
assume ~22/s). Illustrative full design: Arm B 5 runs x 68,544 = 342,720 prompts; Arm A 6
vintages x 12 templates x 6 rho x 80 paths x 2 = 69,120; crossed block ~5,760; gate 26 x 700
= 18,200: about 436,000 prompts, about 5 hours on a 4090 (about $4.50) plus model loads. Compute
is not the binding constraint under the $50 cap: Arm A on all 26 vintages, or doubling the
number of paraphrases, both fit.

**Update for templates v2 (Decision 12):** Arm B 5 x 137,088 = 685,440 prompts; Arm A 6 vintages
x 24 x 6 x 80 x 2 = 138,240; crossed block about 11,500; gate about 18,200: about 853,000 prompts,
about 10 hours on an RTX 4090 (about $9) or roughly a third of that time on an H100 at a similar
total. Still well inside the $50 cap.

**Frozen allocation (2026-11-12, `configs/generation.yaml`):** 959,916 prompts in 505 shards
(26 ChronoGPT vintages, of which 1999, 2009 and 2024 run every Arm B quarter, plus 7 ladder
models), K = 32 plus a greedy read. At the pilot's ~25–30 prompts/s on an RTX 4090 this is about
9–11 GPU-hours (about $8–10 at $0.89/hr) plus model downloads and loads; an H100 should take
roughly a third of the time at about $4/hr. Session ceiling proposed in the PAP (A14): $35.
