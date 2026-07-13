# C-REL — Relational-Object gate → **G-R.1 FAIL (sixth clean negative; the floor is object-general)**

*Developer 2. The five completed negatives all scored the **pointwise** per-perturbation delta δ_p
(cross-donor reproducibility approximately 0.03). This gate tested a **different object**: the *relational*
structure over perturbations (similarity S, program loadings L, per-gene rank R), which averages over
many genes so that per-cell noise averages out. The question is whether the **specific**
(shared-program-removed) relational structure is recoverable where the pointwise delta is not. The
analysis is CPU-only, with no GPU and no build (G14 fenced). CP2 and the budget remain frozen.*

**Provenance:** [IN-PROJECT] measured here on committed data · [VERIFIED] startup and prior numbers
re-derived from the repository (below) · [INFERENCE] the reframe under test.

## Source-of-truth re-verification (done first, per the brief)
- Startup was confirmed against the repository: `main` `b3581ef` (local==remote), 10 tags including
  `donor-final`, `benchmark_table.csv` byte-identical since `81ef528`, split SHA `fd2b8c21…`.
- The "0.03 pointwise floor" is real: per-perturbation cross-donor δ reproducibility = 0.033
  (`phaseB_snr_precheck` top-50), 0.034 (PHASEB individual-pert), 0.032 (BUDGET specific perm null),
  0.034 (DONOR donor-averaged). One object, approximately 0.03. Re-measured here on the same pipeline,
  the value is 0.049.
- The "0.94" measured the *aggregate* object, not the specific one: `phaseB_localization.csv`
  `profile_corr_within_condition = 0.9408` is the reproducibility of the gene-residual profile
  *averaged over approximately 1300 perturbations* (the shared program); cross-*condition* is only 0.42.
  It was used as motivation only, never as a prediction (the swing-#5 mistake).

## Method (specific space is mandatory)
Per (perturbation × context × donor) committed pseudobulk δ (frozen normalization); specific residual
`δ_specific(p,c,d) = δ(p,c,d) − mean_{p'} δ(p',c,d)` per (condition, donor); pooled over conditions;
1,926 perturbations common to all 4 donors. The objects on the specific residuals are: **S**, the
pert×pert cosine similarity (reproducibility = cross-donor Pearson of the similarity pattern); **L**, the
SVD gene factors plus per-donor per-pert scores; and **R**, the per-gene rank order of perturbations.
The baselines are the 0.03 pointwise floor (same data) and a label-permutation null. The pass criterion
is: at least 1 specific object at or above 0.30, above null.

## Result — FAIL (no object reaches 0.30)

| object | specific-space | raw contrast | perm null |
|---|---|---|---|
| S — similarity (all 1,926 perts) | **0.008** | 0.007 | 0.000 |
| S — top-200 high-effect perts | **0.037** | — | 0.001 |
| L — loadings (top-3; factor-1) | **0.111** (0.17) | — | 0.019 |
| R — per-gene rank | **0.025** | 0.024 | — |
| pointwise floor (ref) | 0.049 | — | — |

The machinery is calibrated: it reproduces the committed pointwise floor (0.049), and it does detect
real structure when present, with L factor-1 reproducing at 0.17, clearly above its 0.019 null.
Nevertheless, no relational object clears 0.30, and the best value (L factor-1 = 0.17) is a genuine but
minute residual (strong-perturbation directions surviving weakly in the top loading), far below the bar.

The floor is total, not the product of population dilution. Restricting S to the top-200 high-effect
perturbations lifts it only to 0.037; even the strongest perturbations' relational similarity does not
reproduce across donors. Relational aggregation over a population that is individually noise-floored does
not rescue it.

## Reported discrepancy vs the brief (repo/measurement wins)
The brief expected raw-space S to reproduce at approximately 0.9 (a shared-program tautology to be
demoted). As measured, it is 0.007, the *same* as the specific space. The reason is that cross-donor
reproducibility of the similarity **pattern** (mean-removed Pearson of cosine similarities) isolates the
reproducible *directional* structure, which is the specific/noise part; the approximately 0.9 the author
had in mind is the trivially high **constant baseline** of raw cosines (all perturbations point toward
the shared program), which is not a reproducible *pattern*. Consequently, even the anticipated tautology
is not one under the correct measure, yielding a cleaner negative than the brief anticipated. Per the
source-of-truth rule, this is reported as measured and is not tuned toward the brief's number.

## Verdict & routing
**G-R.1 FAIL → the floor is object-general → sixth clean negative → STOP.** G-R.2 (known-biology
recovery vs a degree-preserving null) is gated on G-R.1 passing and was not run, because there is no
reproducible specific-relational structure to test for biology. The relational-JEPA build (G14) is not
licensed; no GPU was spent; nothing was merged.

The per-perturbation frontier is now floored across six independent objects and methods: causal-matrix,
fluctuation/third-moment, single-cell SNR, trajectory-geometry, donor-structure, and now relational
structure — pointwise *and* relational, raw *and* specific, whole-population *and* high-effect subset.
The submission's noise-floor conclusion holds and is strengthened.

## Honest ceilings (preserved)
- The single whisker of signal (L factor-1 = 0.17) is real but tiny and below bar; it does not license a
  build. It should not be over-read (the 0.48 / 0.94 lesson).
- A genuinely notable relational result would be **cross-dataset transfer** (relations transfer where
  absolute deltas do not); this is a separate, later project, not this gate, and it is not evidenced here.
- No method-novelty claim is made: relational/contrastive perturbation modeling is an active area
  (Shesha, GEARS, contrastive-perturbation methods; Marson's own regulator→program map is a relational
  analysis). Occupancy is a hard pre-manuscript check, not asserted from memory.

Outputs: `results/relational_gate.csv`, `figures/relational_gate.png`.
