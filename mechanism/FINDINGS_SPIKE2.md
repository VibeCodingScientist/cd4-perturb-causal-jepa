# Spike #2 — Does the mechanism win once the mechanism is nonlinear?

**Verdict: FAIL / PARK** (a clean confirmed negative). This result is handed off to the project lead.

## Summary of findings

The hypothesis was that nonlinearity combined with combinatorial perturbations would dissolve the two
advantages that correlation enjoys in the linear regime (namely, covariance as an approximately
sufficient statistic for `A`, and coverage of never-perturbed genes), thereby allowing explicit
mechanism estimation to finally overtake correlation on held-out **double**-perturbation
transportability. It does not. Across the full nonlinearity grid `λ ∈ {0, 0.25, 0.5, 0.7, 0.85}`, the
mechanism−correlation gap remains flat and slightly negative (−0.003 → −0.001 pooled AUROC, with every
confidence interval hugging or spanning 0); it never crosses zero. The manipulation itself was
effective: mean relative epistasis rose 0 → 0.18, the fraction of transportable doubles fell
0.50 → 0.30 (under saturation, even the basal shift of mode `b` ceases to be transportable, exactly as
predicted), and the linear oracle degraded 1.00 → 0.88. This is therefore a genuine nonlinear regime
rather than a failed knob. Because the binary AUROC is ceiling-saturated (approximately 0.95–1.0 for
all methods), the finer continuous metric was examined: the Spearman gap in fact shrinks with λ
(+0.016 at λ=0 → −0.027 at λ=0.85), meaning that the mechanism *loses* ground as nonlinearity grows,
the opposite of the hypothesis. The decisive diagnostic is effect capture. Simply summing the
**observed** singles predicts the true double effect at cosine 0.99 → 0.98 even at epi_rel 0.18,
whereas the mechanism sits at 0.69 → 0.64 and degrades; its `Â`-estimation error (approximately 20%,
inherited from spike #1's P≪G wall) swamps the approximately 18% epistasis signal it is attempting to
exploit. Additivity of directly-observed singles is thus remarkably robust for the *direction* (and
hence the transportability) of a double, and a mechanistic model built on an imperfect `Â` cannot beat
it even where epistasis is genuinely present. The case `λ=1.0` is degenerate (the pure-saturation
Jacobian collapses; 32/32 instances non-Hurwitz) and is dropped. In sum, correlation/additivity
remains robust even under nonlinearity with this estimator — a clean negative that sharpens, rather
than overturns, spike #1.

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

## Why the mechanism does not win — supplementary (`results/spike2_diagnostics.csv`)

**(1) The finer metric rules out the AUROC ceiling.** Spearman(score, continuous agreement) has
headroom, and the mech−corr gap trends in the *wrong* direction with λ:

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

Summing observed singles recovers the double's *direction* nearly perfectly even with 18% epistasis;
the mechanism's imperfect `Â` performs far worse and degrades with λ. The epistasis is real in
*magnitude* but does not rotate the effect enough to matter for cosine-based transportability, and the
`Â`-estimation error dominates whatever epistasis-capture the nonlinear solve provides.

## A standalone positive result (worth retaining regardless)

The **linear transportability condition drifts from truth as biology becomes nonlinear**: the linear
oracle (`−A_true⁻¹Γ`, operating-point-blind) holds AUROC 1.0 through λ=0.5 but falls to 0.876 at
λ=0.85. The field's linear/additive transportability assumption is a good approximation only in the
weakly-nonlinear regime; it loses approximately 12 AUROC points by strong saturation. That degradation
curve is itself a result.

## Kill-probe decision trace (M0 / M1)

- **M0 anchor (λ=0):** gap = −0.001, epistasis = 0.0, reproducing spike #1's regime (correlation not
  beaten; additive) ✓. Note that at λ=0 the double task is *ceiling-easy* (all methods approximately
  1.0) because the test doubles are pairs of individually-perturbed genes: the coverage fix works, but
  there is no headroom for a linear-regime margin, so the anchor is a tie-at-ceiling rather than a wide
  correlation win.
- **M1 movement (λ=0.85):** the gap moved +0.005 relative to λ=0, which is below the +0.10 bar; the
  marginal "cross" (+0.004, CI includes 0) is noise. Strictly, this is a **PARK**. The full grid was
  run regardless to produce a complete curve, and it confirms the flat trend.

## Parameters and deviations (documented, not tuned on labels)

Carried over from spike #1 unchanged: explicit estimation only (no attention), Lyapunov `−D` sign fix,
identity-fill for unidentified `Â` rows, `alpha=0.002`, `n_cells=1000`, `G=50, n_reg=6, P_train=30`.
New for spike #2: `h_λ(x)=(1−λ)x+λ·s·tanh(x/s)`; the test comprises 25 held-out doubles of
individually-perturbed genes per instance; the mechanism estimates `Â` from the **nonlinear**
interventional constraint `A·(h_λ(x_pert)−h_λ(x_ctrl)) = −Γ` (reusing `estimate_A`, which reduces to
spike #1 at λ=0) and predicts doubles by solving the nonlinear fixed point with the *known* λ, s. The
**saturation scale is `s=0.4`** (not the brief's illustrative 1.5): a manipulation check showed that
at `s=1.5` the small operating point (|x*|≈0.24) keeps tanh in its linear zone and induces
approximately 0 epistasis (a false PARK), whereas `s=0.4` yields a clean epistasis gradient 0→0.18
while remaining Hurwitz through λ=0.85. Observation model: latent pseudobulk (finite-cell mean around
the fixed point with local-Jacobian covariance) rather than the NB/log-count emission — required for
the nonlinear fixed-point machinery to be self-consistent at order-1 latent magnitudes. The
correlation null still uses the local control covariance, preserving its linear-regime edge so that
the λ=0 anchor remains valid.

## Reproduce

```bash
python run_spike2.py     # kill probe -> (if it moves) full grid -> results/gap_vs_lambda.{csv,png}
python spike2_diag.py    # supplementary Spearman + effect-capture -> results/spike2_diagnostics.csv
```
CPU-only, a few minutes.

## Bottom line for the project lead

Two independent spikes now agree: the transportability signal is real (spike #1 oracle = 1.0), but an
**explicitly-estimated per-context `Â` does not beat correlation/additivity** — neither under P≪G
linearity (spike #1) nor under nonlinearity/epistasis with this estimator (spike #2). The blocker is
`Â` quality, and nonlinearity does not relax it. Should this line be pursued, the lever is a
**materially better `A` estimator** (or an intervention design that makes correlation/additivity fail
harder than `Â`-estimation does), not more nonlinearity. Out of scope by rule: real ZHU25 data, A/B
decomposition, acquisition, and the if-time estimator upgrade (the gap did not cross, so it was not
built).
