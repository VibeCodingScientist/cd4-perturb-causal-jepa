# Pre-registration — CD4+ T Cell Perturbation Prediction

**Committed BEFORE any model saw data.** (`UNIFIED_BUILD_PLAN.md` §2.) The split
(`split_manifest.json`) is frozen and SHA-bound to the dataset; the interpretations
below are written now so analysis is a look-up, not a scramble.

Dataset: Marson/Pritchard genome-scale CRISPRi Perturb-seq in primary human CD4+ T
cells (~22M cells; every expressed gene silenced one at a time; 4 donors; 3
activation states Rest / Stim8hr / Stim48hr). GEO **GSE278572**.

Primary test regime: the **condition hold-out** — full **Stim48hr** withheld, train
on Rest + Stim8hr, evaluate zero-shot on Stim48hr. Secondary: **gene hold-out** (15%
of perturbed genes withheld; interpolation). Sanity: **donor probe** (donor_4).

---

## Claims

- **C1 (causal, external).** The `CausalCisTransFormer` with the corrected do-mask
  **matches or beats strong baselines including TabPFN** on the condition hold-out
  (zero-shot Stim48hr), measured by Pearson-delta on the top-50 DEGs.

- **C2 (do-operator isolation).** The causal mask **beats its non-causal twin** on
  the condition hold-out (identical architecture, mask on vs off — the single most
  important ablation). Reported **regardless of leaderboard position**.

- **C3 (JEPA).** **JEPA-init helps** the condition hold-out. Evaluated as the full
  2×2 (encoder-init × causal-mask).

- **S1 (support / VOI).** Ensemble model-disagreement **ranks which perturbations
  are most worth measuring**; VOI-guided selection reaches **~90% of full-screen
  accuracy from a fraction** of perturbations.

## The 2×2 (experimental core)

| Encoder init | Causal mask | Model name | Role |
|---|---|---|---|
| Random | off | `noncausal`    | Direct-regression baseline |
| Random | on  | `causal`       | Causal-only (**C2 treatment**) |
| JEPA   | off | `jepa_only`    | JEPA-only |
| JEPA   | on  | `jepa_causal`  | **JEPA + causal (main model)** |

The causal-vs-noncausal contrast (**C2**) and the JEPA-init contrast (**C3**) both
read directly off this matrix on condition-hold-out Pearson-delta (Figure 2).

## Pre-committed outcome interpretations

| Outcome | Interpretation |
|---|---|
| Causal beats non-causal on condition hold-out (any init) | do-operator provides real inductive bias under activation-state shift. **Headline.** |
| JEPA-init beats random-init on condition hold-out | representation robustness transfers to cross-context perturbation delta — **extends Cell-JEPA** (which only tested within one cell line). Report prominently. |
| JEPA helps absolute-state but not gene-hold-out delta | **corroborates Cell-JEPA's "complementary aspects" finding.** Clean positive-or-null, still publishable. |
| TabPFN wins gene hold-out; causal wins condition hold-out | strongest scientific result: **priors matter for distribution shift, not interpolation.** |
| TabPFN wins both | tabular ICL beats specialized biology architectures on primary immune cells — **publishable negative.** |

**Grounding for C3.** Cell-JEPA (arXiv 2602.02093) reports that within a single cell
line JEPA improves absolute-state reconstruction but **not** effect-size (delta)
estimation, and did not test cross-context transfer. The condition hold-out is
therefore the untested regime: a null there **corroborates**, a positive there is a
genuine **extension**. Either way the 2×2 answers a real question.

## Two corrections this build will not revert (`UNIFIED_BUILD_PLAN.md` §1)

1. **Do-mask propagates.** An intervention removes only edges *into* the perturbed
   gene (mask its query row); other genes must still attend to it so the effect
   propagates downstream. We do **not** add `M[:, perturbed] = -inf`.
   (DoFormer, bioRxiv 2026.05.02.722054.)
2. **JEPA = EMA teacher at single-cell resolution.** Student(masked) predicts a
   stop-gradient EMA teacher(unmasked) via a predictor head + cosine loss, masking
   expression *values* within a cell — not a pseudobulk MLP.
   (Cell-JEPA, arXiv 2602.02093.)

## Evaluation (frozen; `core/eval.py`)

- **Headline (demo):** Pearson-delta on top-50 DEGs; PerturBench rank /
  perturbation-discrimination (**mode-collapse detector** — any model > 0.4 flagged
  red); DES (sign-correct DEG overlap). Reported separately for gene and condition
  hold-out.
- **Appendix (auto, never in demo):** + MAE, Spearman LFC, Spearman effect-size,
  AUPRC, E-distance.
- The mode-collapse detector is essential: models that "win" MAE by predicting the
  control/mean are surfaced explicitly.

---

## C-TC — Trajectory-Coupling (pre-registered 2026-07-09; gate-tested same night)

Hypothesis: predictability is a *dynamical* property — unrecoverable perturbations move the cell
*along* the activation trajectory (a fixed axis `a` = Rest→Stim48hr control shift) rather than
shifting an endpoint, so the identifiable target is a scalar displacement `s_p = proj(δ_p, a)`.

- **C-TC.1 (geometry).** Recoverability `R_p` (do-operator `causal_frac_of_ceiling`) decreases as
  trajectory-coupling `TC_p = |proj(δ_p,a)|/‖δ_p‖` increases.
  *Pre-registered PASS:* |partial Spearman(R_p, TC_p | ‖δ‖, reliability)| ≥ 0.30, p < 0.01, sign
  consistent ≥ 3/4 donors. **RESULT: FAIL** — partial ρ = +0.007 (condition), +0.034 (gene);
  p = 0.75 / 0.55; 1D and 2D activation subspace both ≈ 0.

- **C-TC.2 (build).** A predictor of `s_p`, reconstructed as `δ̂ = ŝ·a`, beats the do-operator on
  high-TC gene-holdout perturbations, with the win from per-perturbation variance of `ŝ` (not the
  shared mean; mode-collapse-guarded). *Gated on C-TC.1 passing AND `s_p` reproducing above the ~0.03
  floor (G-TC.0).* **NOT RUN** — G-TC.0 found `s_p` cross-donor reproducibility ~0.07 (≈ random-axis
  null 0.03), i.e. at the floor, and C-TC.1 failed. No build; no GPU spent.

**Conclusion:** predictability is *not* a trajectory-geometry property on this data.
