# Spike #2 — Does the mechanism win once the mechanism is nonlinear?

**Verdict: FAIL / PARK** (clean confirmed negative). Hand off to the project lead.

## The readout (~10 sentences)

The hypothesis was that nonlinearity + combinatorial perturbations would dissolve correlation's two
linear-regime advantages (covariance ≈ sufficient statistic for `A`; coverage of never-perturbed
genes), letting explicit mechanism estimation finally overtake it on held-out **double**-perturbation
transportability. It does not. Across the full nonlinearity grid `λ ∈ {0, 0.25, 0.5, 0.7, 0.85}` the
mechanism−correlation gap **stays flat and slightly negative** (−0.003 → −0.001 pooled AUROC, every
CI hugging or spanning 0); it never crosses zero. The manipulation itself worked — mean relative
epistasis rose 0 → 0.18, the fraction of transportable doubles fell 0.50 → 0.30 (under saturation
even mode `b`'s basal shift stops being transportable, exactly as predicted), and the linear oracle
degraded 1.00 → 0.88 — so this is a real nonlinear regime, not a failed knob. The binary AUROC is
ceiling-saturated (~0.95–1.0 for everyone), so we checked the finer continuous metric: the Spearman
gap actually **shrinks** with λ (+0.016 at λ=0 → −0.027 at λ=0.85) — the mechanism *loses* ground as
nonlinearity grows, the opposite of the hypothesis. The decisive diagnostic is effect capture: simply
summing the **observed** singles predicts the true double effect at cosine **0.99 → 0.98** even at
epi_rel 0.18, while the mechanism sits at **0.69 → 0.64 and degrades** — its `Â`-estimation error
(~20%, inherited from spike #1's P≪G wall) swamps the ~18% epistasis signal it is trying to exploit.
So additivity of directly-observed singles is remarkably robust for the *direction* (hence
transportability) of a double, and a mechanistic model built on an imperfect `Â` cannot beat it even
where epistasis is genuinely present. `λ=1.0` is degenerate (the pure-saturation Jacobian collapses;
32/32 instances non-Hurwitz) and is dropped. Net: correlation/additivity is robust even under
nonlinearity with this estimator — a clean negative that sharpens, rather than overturns, spike #1.

## Headline curve — `results/gap_vs_lambda.{csv,png}`

Pooled AUROC (transportable vs blocked doubles), 8 seeds/mode × 25 pairs = 800 doubles per λ.

| λ | epistasis (rel) | frac transportable | mechanism | corr-add null | GEARS-add null | obs-add null | linear oracle | **gap (mech−corr)** [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.50 | 0.997 | 1.000 | 0.993 | 1.000 | 1.000 | **−0.003** [−0.007, −0.001] |
| 0.25 | 0.024 | 0.50 | 0.996 | 1.000 | 0.995 | 1.000 | 1.000 | **−0.004** [−0.009, −0.000] |
| 0.50 | 0.062 | 0.50 | 0.988 | 0.998 | 0.997 | 1.000 | 1.000 | **−0.010** [−0.023, −0.001] |
| 0.70 | 0.113 | 0.44 | 0.971 | 0.982 | 0.961 | 0.996 | 0.956 | **−0.011** [−0.026, +0.005] |
| 0.85 | 0.182 | 0.30 | 0.982 | 0.983 | 0.946 | 0.999 | 0.876 | **−0.001** [−0.012, +0.017] |
| 1.00 | — | — | — | — | — | — | — | degenerate: 32/32 unstable (dropped) |

## Why the mechanism doesn't win — supplementary (`results/spike2_diagnostics.csv`)

**(1) Finer metric rules out the AUROC ceiling.** Spearman(score, continuous agreement) has headroom,
and the mech−corr gap trends the *wrong* way with λ:

| λ | Spearman mech | Spearman corr | Spearman obs | **Spearman gap (mech−corr)** |
|---|---|---|---|---|
| 0.00 | 0.804 | 0.788 | 0.896 | **+0.016** |
| 0.25 | 0.839 | 0.801 | 0.909 | **+0.038** |
| 0.50 | 0.879 | 0.863 | 0.972 | **+0.016** |
| 0.70 | 0.880 | 0.883 | 0.981 | **−0.003** |
| 0.85 | 0.843 | 0.870 | 0.977 | **−0.027** |

**(2) Effect capture — the crux.** Mean `cos(predicted double effect, TRUE double effect)`:

| λ | epistasis (rel) | mechanism | **obs-add (sum of observed singles)** | corr-add |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.693 | **0.991** | 0.559 |
| 0.50 | 0.062 | 0.681 | **0.991** | 0.549 |
| 0.85 | 0.182 | 0.640 | **0.978** | 0.481 |

Summing observed singles nearly perfectly recovers the double's *direction* even with 18% epistasis;
the mechanism's imperfect `Â` does far worse and degrades with λ. The epistasis is real in *magnitude*
but does not rotate the effect enough to matter for cosine-based transportability, and the
`Â`-estimation error dominates whatever epistasis-capture the nonlinear solve provides.

## A standalone positive result (worth keeping regardless)

The **linear transportability condition drifts from truth as biology becomes nonlinear**: the linear
oracle (`−A_true⁻¹Γ`, operating-point-blind) holds AUROC 1.0 through λ=0.5 but falls to **0.876 at
λ=0.85**. The field's linear/additive transportability assumption is a good approximation only in the
weakly-nonlinear regime; it loses ~12 AUROC points by strong saturation. That degradation curve is
itself a result.

## Kill-probe decision trace (M0 / M1)

- **M0 anchor (λ=0):** gap = −0.001, epistasis = 0.0 → reproduces spike #1's regime (correlation not
  beaten; additive) ✓. Note: at λ=0 the double task is *ceiling-easy* (all methods ~1.0) because test
  doubles are pairs of individually-perturbed genes — the coverage fix works, but there is no headroom
  for a linear-regime margin, so the anchor is a tie-at-ceiling rather than a wide correlation win.
- **M1 movement (λ=0.85):** gap moved +0.005 vs λ=0 — **below the +0.10 bar**; the marginal "cross"
  (+0.004, CI includes 0) is noise. Strictly this is a **PARK**. The full grid was run anyway for a
  complete curve and confirms FLAT.

## Parameters & deviations (documented, not tuned on labels)

Carried from spike #1 unchanged: explicit estimation only (no attention), Lyapunov `−D` sign fix,
identity-fill for unidentified `Â` rows, `alpha=0.002`, `n_cells=1000`, `G=50, n_reg=6, P_train=30`.
New for spike #2: `h_λ(x)=(1−λ)x+λ·s·tanh(x/s)`; test = 25 held-out doubles of individually-perturbed
genes per instance; mechanism estimates `Â` from the **nonlinear** interventional constraint
`A·(h_λ(x_pert)−h_λ(x_ctrl)) = −Γ` (reuses `estimate_A`; reduces to spike #1 at λ=0) and predicts
doubles by solving the nonlinear fixed point with the *known* λ, s. **Saturation scale `s=0.4`** (not
the brief's illustrative 1.5): a manipulation check showed that at `s=1.5` the small operating point
(|x*|≈0.24) keeps tanh in its linear zone and induces ~0 epistasis (a false PARK); `s=0.4` yields a
clean epistasis gradient 0→0.18 while staying Hurwitz through λ=0.85. Observation model: latent
pseudobulk (finite-cell mean around the fixed point with local-Jacobian covariance) rather than the
NB/log-count emission — required for the nonlinear fixed-point machinery to be self-consistent at
order-1 latent magnitudes; the correlation null still uses the local control covariance, preserving
its linear-regime edge so the λ=0 anchor is valid.

## Reproduce

```bash
python run_spike2.py     # kill probe -> (if it moves) full grid -> results/gap_vs_lambda.{csv,png}
python spike2_diag.py    # supplementary Spearman + effect-capture -> results/spike2_diagnostics.csv
```
CPU-only, a few minutes.

## Bottom line for the project lead

Two independent spikes now agree: the transportability signal is real (spike #1 oracle = 1.0), but an
**explicitly-estimated per-context `Â` does not beat correlation/additivity** — not under P≪G linearity
(spike #1), and not under nonlinearity/epistasis with this estimator (spike #2). The blocker is `Â`
quality, and nonlinearity does not relax it. If this line is pursued, the lever is a **materially
better `A` estimator** (or an intervention design that makes correlation/additivity fail harder than
`Â`-estimation does) — not more nonlinearity. Out of scope by rule: real ZHU25 data, A/B decomposition,
acquisition, the if-time estimator upgrade (the gap did not cross, so it was not built).
