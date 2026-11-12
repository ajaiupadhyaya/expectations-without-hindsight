# Vendored ChronoGPT code

- `ChronoGPT_instruct.py`, `config.json`, `requirements.txt`, `MODEL_CARD_1999.md`: from
  `manelalab/chrono-gpt-instruct-v1-19991231` at the revision pinned in `configs/generation.yaml`.
  `ChronoGPT_instruct.py` is byte-identical in all 26 instruct repos (checked 2026-11-11).
- `base_ChronoGPT_inference.py`: from `manelalab/chrono-gpt-v1-19991231` (pilot model).
- Licence: MIT (model cards). Authors: He, Lv, Manela and Wu.
- Not edited. Our batched, cached sampler (`ewh/models/chrono_fast.py`) re-implements the forward
  pass and is tested against `ChronoGPT.forward` from this file.
