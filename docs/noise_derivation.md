# How forecast noise biases the error-on-revision coefficient, and how to remove it

Phase 1 deliverable (plan Task 1.4). Every claim below is checked by a test in
`tests/test_estimators.py` (named in brackets) and illustrated by `make noise-mc`.

## 1. Setup

A unit i is one revision pair: one path (or SPF quarter), one prompt template, one model.
The forecaster's systematic forecasts of the same target x = x_{t+1} are F\*\_{t-1} (made at
t−1) and F\*\_t (made at t). Write

- systematic error e\* = x − F\*\_t, systematic revision r\* = F\*\_t − F\*\_{t-1};
- C = Cov(e\*, r\*), V = Var(r\*), and the noise-free coefficient b\* = C / V.

Each observed draw adds noise: f\_{t,k} = F\*\_t + u\_{t,k}, f\_{t-1,k} = F\*\_{t-1} + u\_{t-1,k},
with Var(u\_t) = σ²\_t, Var(u\_{t-1}) = σ²\_{t-1}, independent across draws, across the two
dates, and of (x, F\*). Independence across dates requires that the two prompts do not share
random numbers (plan D3: the stream is keyed on the prompt id). Cell intercepts (one per rho,
or per variable) are handled by demeaning within the cell throughout.

## 2. The one-draw coefficient

**Proposition 1.** With one draw per prompt, error e = e\* − u\_t and revision
r = r\* + u\_t − u\_{t-1}, so

    plim b_naive = (C − σ²_t) / (V + σ²_t + σ²_{t-1}).

*Proof.* Cov(e, r) = C + Cov(−u\_t, u\_t) = C − σ²\_t, because every other cross term is zero
by independence. Var(r) = V + σ²\_t + σ²\_{t-1}. ∎ [`test_naive_matches_general_formula`]

The handoff's formula (Section 3.1) is the special case σ²\_t = σ²\_{t-1}. In our design the
two forecasts are at different horizons (two steps at t−1, one step at t), so equal variances
should not be assumed.

**Corollary (attraction point).** Let S = σ²\_t + σ²\_{t-1}, w = V/(V + S) (the signal share
of revision variance) and b° = −σ²\_t / S. Then

    plim b_naive = w · b* + (1 − w) · b°.

The one-draw coefficient is a weighted average of the truth and the attraction point b°,
which is −1/2 when the two noise variances are equal. Noise therefore pulls the coefficient
toward b° *from either side*: a rational forecaster (b\* = 0) or an underreactor appears to
overreact, but a strong overreactor (b\* < b°) appears to overreact *less* than it does.
"Noise inflates measured overreaction" is true only for b\* > b°. H1 is therefore tested
two-sided (plan change 2). [`test_attraction_point`]

This matters for the literature: Gao, Jiang and Yan report b ≈ −0.46 at rho = 0 for
Qwen3-32B and AKLMT's human participants give about −0.48 (our reproduction of their data,
`data/processed/aklmt_figure2.csv`). Both sit near the attraction point, which is where a
noisy forecaster lands whatever its systematic coefficient.

## 3. Averaging k draws: the noise ladder

Averaging k independent draws divides each noise variance by k:

    b(k) = (C − σ²_t / k) / (V + S / k),

so b(1) is the one-draw limit and b(k) → b\* as k → ∞, with the gap shrinking like 1/k.
Plotting b(k) against k (the handoff's "noise ladder") shows how much of a measured
coefficient is noise. [`test_mean_of_k_tracks_b_of_k`]

## 4. Split-sample IV

Split each prompt's draws into disjoint halves A and B. Let y = x − F\_t^A,
r\_A = F\_t^A − F\_{t-1}^A and z = r\_B = F\_t^B − F\_{t-1}^B. The noise in half B is independent
of the noise in half A and of (x, F\*), so

    Cov(z, y) = Cov(r* + noise_B, e* − ū_t^A) = C,   Cov(z, r_A) = Cov(r* + noise_B, r* + noise_A) = V,

and the IV estimator Cov(z, y)/Cov(z, r\_A) is consistent for b\* for any noise level.

## 5. Averaging over every split, in closed form

One random split wastes information and makes the estimate depend on the split. Averaging the
IV numerator and denominator over all balanced splits has an exact closed form (plan change 8).
For a unit with K draws at a date, mean F̄ and sample variance s² (ddof 1), the finite-
population identities for halves of size K/2 are

    E_split[F^A] = F̄,   E_split[(F^A)²] = F̄² + s²/K,   E_split[F^A F^B] = F̄² − s²/K.

Splits at the two dates are independent, so with ē = x − F̄\_t and r̄ = F̄\_t − F̄\_{t-1}:

    E_split[z · y]   = ē r̄ + s²_t / K_t                      =: N_i
    E_split[z · r_A] = r̄² − s²_t / K_t − s²_{t-1} / K_{t-1}  =: D_i

With cell intercepts the IV moments are Σ(z − z̄\_c)(y − ȳ\_c). Splits are independent across
units, so averaging gives, within a cell of n\_c units,

    Σ (ē_i − ē_c)(r̄_i − r̄_c) + (1 − 1/n_c) Σ s²_{t,i}/K_{t,i}

and the analogous denominator. The primary estimator is the ratio of these two sums
(`ewh.estimators.eor.split_iv`). It equals the all-splits average exactly
[`test_split_iv_equals_enumeration` enumerates every split; `test_cross_unit_averaging_by_
brute_force` checks the cross-unit step over all joint splits of two units, to 1e-10].

The same expression with s²/K replaced by s²/K − s²/k gives b(k) for every k
(`b_of_k`), so the one-draw, mean-of-k and IV estimators are one family on one common
sample [`test_common_sample`].

## 6. Inference

The estimator solves Σ(a\_i − b·d\_i) = 0, where a\_i and d\_i are a unit's numerator and
denominator contributions. Its influence value is ψ\_i = (a\_i − b d\_i)/mean(d), and
Var(b̂) = Var(mean ψ) under the chosen clustering (paths, two-way, Driscoll–Kraay). The
numerator's own noise (the estimated s²) enters ψ, so no separate correction is needed.
Standard errors match statsmodels exactly up to its stated small-sample factors
[`tests/test_inference.py`].

The IV is weak when the noise-free revision variance is small relative to its sampling error:
at low rho for a near-rational forecaster, the revision is close to zero. Every estimate
therefore reports the first-stage strength t² = mean(d)²/Var(mean d) and the Fieller set
{b₀ : (ā − b₀ d̄)² ≤ χ²₁,₀.₉₅ Var(ā − b₀ d̄)}, which may be unbounded
[`test_weak_first_stage_flagged`]. Coverage of the 95% interval is 93–97% over 2,000
replications for every combination of planted coefficient in {−0.7, −0.3, 0, 0.3} and
noise share in {0, 0.3, 0.6} [`test_split_iv_coverage`, the handoff's required test].

## 7. What the correction does and does not remove

- It removes **sampling noise**: variation between draws from the same prompt.
- It does **not** remove noise that is common to all draws of a prompt: the effect of the
  particular wording, number format or path. That variation is part of F\* for that unit.
  The cross-template instrument (plan D5, crossed block) addresses wording noise by
  instrumenting with other paraphrases on the same path. This is why Gao, Jiang and Yan can
  find b < 0 at temperature 0, where sampling noise is zero.
- Invalid draws are dropped by an outcome-independent rule (plan D6), so the estimand is the
  coefficient on means of draws truncated to a band around the last shown value. The paper
  states this.

## 8. Benchmarks are not all zero

The known-parameter rational forecaster has b\* = 0 for rho > 0. The least-squares learner,
which fits an AR(1) to exactly the 40 values shown, does not: its coefficient is clearly
negative at low persistence and positive at high persistence, while its implied persistence
is close to unbiased at every rho [`test_ls_learner_benchmark_values`; values in
`docs/decisions.md`, Decision 4]. A model's b is therefore compared both with zero and with
this learner on the same paths (plan change 1, H2-LS).
