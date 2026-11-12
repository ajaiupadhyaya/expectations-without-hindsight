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
