# Pilot log

Every pilot run before the freeze: date, model (must be outside every confirmatory set),
seeds (pilot namespace only), rho (0.5 only), what was measured, and what it was used for.
Pilots tune parsing, throughput and power assumptions, nothing else.

## Pilot 1 (2026-11-11): Qwen/Qwen2-0.5B-Instruct on this Mac — power calibration
- Model: `Qwen/Qwen2-0.5B-Instruct` @ c540970f (parsing stand-in; not in the ladder, not a
  ChronoGPT model). Pilot seeds (namespace `pilot`), rho = 0.5 only, all 12 templates, 20 paths
  per template (480 prompts), K = 16 plus a greedy read, temperature 1, chat template.
- A first attempt on CPU (40 paths) was stopped after one shard because it was too slow
  (about 10 s per prompt); its partial draws are kept in `data/pilot/qwen2-0.5b/` and unused.
- Results (`data/pilot/qwen2-0.5b-mps20/results.json`, MPS GPU, 0.8 prompts/s):
  - valid draws 91% (this model would fail the 95% parse gate);
  - within-prompt draw variance is 96% (date t) and 90% (date t−1) of all variation in single
    draws; per-draw noise s.d. is about 1.6 innovation s.d. (kappa ≈ 1.6);
  - one-draw coefficient −0.52 (s.e. 0.02), essentially the attraction point −1/2 that pure
    noise produces (docs/noise_derivation.md); split-sample IV −0.19 (s.e. 0.73), weakly
    identified with 239 units;
  - per-paraphrase IV estimates are unstable (40 units each), so the spread across paraphrases
    (tau) cannot be estimated from this model: tau-hat = 0 because sampling noise dominates.
- Used for: the noise level in the power grid (kappa between 1 and 2 is realistic for a small
  model); confirming that template inference must rest on many units per paraphrase.
- Not used for: any hypothesis, threshold or template choice.

## Pilot 2 (2026-11-12): ChronoGPT base model on RunPod — sampler equivalence and throughput
- Model: `manelalab/chrono-gpt-v1-19991231` @ 45d65683 (NON-instruct base; outside every
  confirmatory set). Pilot seeds, rho = 0.5, the 480-prompt pilot table. RTX 4090 24 GB, Secure
  Cloud EU-RO-1, pod `r5zmy0q27169tc`, torch 2.15.0+cu132, driver 595.91.07, CUDA 13.2. AJ
  approved the run and its price ($0.89/hr) on 2026-11-11.
- Only sampler-equivalence, parse-rate, throughput and load-time statistics were computed (plan
  global constraint); no forecast or gate statistic. Raw outputs: `data/pilot/chrono-base-gpu/`.
- Operational problems, all in my pod scripts, fixed during the run: the image blocks
  system-wide pip (uv now installs into its own venv); `config.pt` carries an extra `model_type`
  key the authors' constructor rejects (config filtered to the four architecture keys); one
  comparison batch of 200 prompts ran out of the 24 GB while the fp32 reference model was still
  loaded (reference freed first, batches of 25).
- **Equivalence against the authors' code (full weights, bf16):**
  - prompt pass: top-1 token agrees on 20/20 prompts; max KL 0.040;
  - teacher-forced decoding on the authors' greedy continuations: top-1 agrees at 316 of 322
    steps where the authors' top-two margin exceeds 0.1; mean KL 0.008, max 0.040;
  - full 12-token greedy continuations identical for 16/30 prompts; where they differ, the
    first divergence is a bf16 near-tie, after which the texts naturally part (no prompt
    diverges at the first token). The first run's "1/50" was a measurement artefact: our
    sampler stops at the first newline and pads with zeros, the authors' loop always runs 12
    tokens;
  - sampled numbers (one draw per prompt, 200 prompts, different random streams): two-sample KS
    p = 0.95; parse rates 58% (authors' sampler) vs 67.5% (ours), a borderline difference
    (z about 2) on a non-instruct model that mostly emits non-numeric text.
  - Verdict: equivalent to bf16 tolerance. Exact greedy identity is not attainable in bf16.
- **Throughput (prompts per second, Arm A-length prompts):** K = 16: 38.7 (batch 16), 41.9 (32),
  31.1 (64); K = 32: 30.7 (16), 29.8 (32), 23.2 (64). Peak GPU memory 19.1 GB. Weights load in
  47 s from the pod's network-backed `/workspace` (first access slower).
- Cost: about 64 minutes of pod time at $0.89/hr, roughly $0.95 (billing read not yet posted).
