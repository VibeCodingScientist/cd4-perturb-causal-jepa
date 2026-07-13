# C4 Mechanism-Recovery Spike — Findings

**Verdict: FAIL** (does not clear the pre-registered bar). The result is handed off to the project lead.

## Summary of the readout

On CausalDGP, an explicitly estimated per-context influence matrix `Â_C` does not recover
the ground-truth transportability label better than the correlation baselines. The best variant of the
mechanism reaches a pooled **AUROC of 0.689 [0.640, 0.732]**, which ties the co-expression
(GEARS-style) null at 0.702 (gap −0.011 [−0.061, +0.037], CI includes 0) and is exceeded by the
correlation-graph null at **0.828** (gap **−0.140 [−0.224, −0.053], CI excludes 0**, in the
adverse direction). The pre-registered PASS required the mechanism to exceed *both* nulls with the gap CI
excluding zero; it exceeds neither, so the result is a FAIL. Importantly, this outcome is not a refutation
of the underlying idea: the oracle that uses the true `A` columns scores an **AUROC of 1.000**, so the
transportability signal is fully present and linearly recoverable provided that `A` is known. The limiting
factor is therefore estimation under P≪G rather than the concept itself. The diagnosis has two components.
First, the row-wise Lasso recovers `A` only weakly from 30 single-node interventions in 50 genes
(`cos(Â,A_true)≈0.29` at the brief's implicit `alpha=0.05`, rising to ≈0.80 at the dev-tuned
`alpha=0.002` but still insufficient). Second, in this linear-Gaussian simulator the control-cell
covariance is a near-sufficient statistic for `A` (the stationary `Σ` solves the Lyapunov equation in `A`),
so the observational correlation null obtains the "did the mechanism change?" signal essentially for free
and sharpens faster than the sparse interventional estimate as sequencing depth grows. The single setting
in which the mechanism adds value over correlation is the pure-rewiring mode `a`, where its graded ranking
(Spearman **0.221 [0.085, 0.398]**, CI excludes 0) beats both nulls (corr 0.066, gears 0.019); however,
this does not carry over to mode `both`, so it constitutes a caveat rather than a pass. Finally, the
brief's exact inversion predictor is pathological for held-out genes, as the rank-deficient `Â` inversion
sign-flips the effect (AUROC 0.624 and a negative mean effect-cosine to truth); the principled column-`k`
predictor reported above is strictly better and represents the mechanism's fairest test.

## Pooled AUROC (median over 8 seeds/mode, 95% cluster-bootstrap CI)

| method | pooled AUROC | 95% CI |
|---|---|---|
| Oracle (true `A` columns) — *reference* | **1.000** | [1.000, 1.000] |
| Null: correlation graph (control covariance) | **0.828** | [0.757, 0.889] |
| Null: co-expression propagation (GEARS-style) | 0.702 | [0.647, 0.748] |
| **Mechanism — `Â` column-k (best variant)** | **0.689** | [0.640, 0.732] |
| Mechanism — `Â` brief inversion predictor | 0.624 | [0.549, 0.695] |

**Mechanism − null gaps (best variant, column-k):**
`− corr_null = −0.140 [−0.224, −0.053]` (CI excludes 0), `− gears_null = −0.011 [−0.061, +0.037]` (tie).

## Per-mode graded ranking — Spearman(score, continuous agreement)

The ranking is defined only where the true agreement varies (the rewiring modes); `none`/`b` have constant
agreement (same `A` ⇒ cos = 1), so Spearman is undefined there.

| mode | mechanism col-k | corr null | gears null | oracle |
|---|---|---|---|---|
| `a` (rewire only) | **0.221 [0.085, 0.398]** | 0.066 | 0.019 | 0.588 |
| `both` (rewire + basal) | 0.062 | **0.278** | 0.028 | 0.572 |

## Why the nulls win (the important caveat for the go/no-go)

- **The label is essentially mechanism-level in this DGP.** Because `none`/`b` share `A`, every perturbation
  is transportable, whereas `a`/`both` change `A` and are therefore blocked. Consequently, "predict
  transportability" collapses largely to "detect that `A` changed," a *global* property. The control
  covariance is an excellent and stable global change-detector, while the sparse per-gene `Â` column is a
  noisy local one.
- **Depth helps the nulls more.** Because `Σ = Lyapunov(A)`, adding cells sharpens the covariance's read
  of `A` faster than it sharpens a P≪G interventional estimate. The correlation null climbs steeply with
  depth while the mechanism remains flat (identifiability-limited rather than noise-limited), so the gap
  *widens*:

  | cells/(ctx,pert) | mechanism col-k | corr null | gears null |
  |---|---|---|---|
  | 500 | 0.696 | 0.719 | 0.625 |
  | 1000 | 0.688 | 0.827 | 0.701 |
  | 2000 | 0.703 | 0.898 | 0.755 |
  | 4000 | 0.719 | 0.959 | 0.870 |

  (`results/sensitivity_ncells.{csv,png}`; oracle = 1.000 at every depth.)
- **Held-out rows are unidentifiable.** `Â` is constrained only along observed intervention directions; a
  gene that is never perturbed has an all-zero estimated row, so the brief's `solve(Â, −Γ)` is dominated by
  the ridge term and sign-flips. The column-k predictor sidesteps this by using the *estimable* target
  column.

## What would change the verdict (for the project lead; not acted on here)

1. **More interventions per context** (P → G), or **multi-gene / combinatorial** perturbations that break the
   covariance ≈ mechanism equivalence — the oracle shows the ceiling is 1.0.
2. **A better `A` estimator** than independent row-wise Lasso (joint sparse+low-rank, or estimating the
   effect map `B=−A⁻¹` directly from observed effect columns).
3. **Non-linear / non-Gaussian regimes** in which the control covariance is *not* a sufficient statistic for
   the mechanism, so that an explicit causal estimate can carry information correlation cannot.

## Milestones

- **M0** simulator verified: empirical pseudobulk effect vs analytic `−A⁻¹Γ`, Pearson **r = 0.959** (>0.9 ✓).
- **M1** estimator + both nulls run end-to-end; all scores in [−1, 1] ✓.
- **M2** full 4 modes × 8 seeds × 15 held-out sweep → `results/c4_auroc.csv`, `results/moneyshot.png` ✓.
- **Done** — this readout.

## Hyperparameters (pre-registered on a disjoint dev-seed set; not tuned after seeing labels)

`G=50, n_reg=6, P_train=30, P_holdout=15, n_cells=1000, sigma=0.5, libsize=1e4, theta=5, mag∈U(1,3),
thresh=0.9, knn=10, ridge=1e-3`, and **`alpha=0.002`** (Lasso). `alpha` was selected on dev seeds
(`seed_base=90000`) by maximizing label-independent `A`-recovery `cos(Â,A_true)`, and *not* by maximizing any
mechanism-vs-null gap; the brief's illustrative `0.05` over-shrinks (`cos≈0.29`, mechanism at chance).
There are two deviations from the brief's literal code, both documented in-source: (a) `n_cells` was changed
from 500 to 1000 so that M0 clears 0.9 with margin; and (b) the Lyapunov call passes `−D`, because the
brief's `solve_lyapunov(A, D)` returns a negative-definite covariance for Hurwitz `A`.

## Reproduce

```bash
python run_c4.py    # M0 check + sweep -> results/c4_records.csv
python eval.py      # -> results/c4_auroc.csv, results/moneyshot.png
python sensitivity.py   # supplementary -> results/sensitivity_ncells.{csv,png}
```
CPU-only; completes in a few minutes.
