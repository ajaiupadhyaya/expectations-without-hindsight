# Compute budget (DRAFT until the GPU pilot; finalized before tag pap-v1)

Hard cap: $50 of RunPod credit in total, storage included (AJ, 2026-11-11).

## Spent so far
| Date | Item | Cost |
|---|---|---|
| — | nothing yet | $0.00 |

## GPU pilot (Task 3.8) — awaiting AJ's go
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
| Arm A main, per vintage | 12 templates x 6 rho x n paths x 2 dates | n from docs/power.md |
| Arm A crossed block | 2 vintages x 12 templates x 6 rho x n_c x 2 | for the cross-template IV |
| Gate, per model | 7 kinds x n_g | |
| Arm B, per full run | 68,544 | 5 runs (annual rolling, six-vintage rolling, 1999, 2009, 2024 throughout) |
| Ladder | one third of Arm A units | descriptive; cut first |
