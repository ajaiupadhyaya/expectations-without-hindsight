# Expectations Without Hindsight

*Noise, capability and bias in language-model forecasts.* Ajai Upadhyaya.

Several papers report that language models over-extrapolate like human survey respondents.
Three things could produce that pattern without any human-like belief formation: the model
has seen the outcome (leakage), random variation in its forecasts mechanically biases the
error-on-revision regression (noise), or a weak model simply echoes the last number
(capability). This project measures expectation bias in language models after removing each
of the three, using vintage models with dated training cutoffs, repeated draws with a
split-sample instrumental-variables estimator, and a pre-registered competence gate.

**Status:** Pre-registered 2026-11-12 (tag `pap-v1`; session binds to `pap-v1.1` after pre-data Deviation 1; plan in `docs/PAP.md`). Phases 0–4 done. Next: the GPU session (Phase 5).

| Document | What it is |
|---|---|
| `docs/HANDOFF.md` | The project specification |
| `docs/PLAN.md` | The approved execution plan |
| `docs/decisions.md`, `docs/assumptions.md` | Running logs of choices and assumptions |
| `docs/novelty_check.md` | Prior work and overlap with the intended contributions |
| `docs/references_verified.md` | Every reference checked against its source |
| `docs/data_terms.md` | Data sources and their terms of use |

```bash
make install
make all
```

Raw downloads are not committed. Each has a sidecar with its source URL, retrieval time and
sha256, indexed in `data/raw/MANIFEST.json`.
