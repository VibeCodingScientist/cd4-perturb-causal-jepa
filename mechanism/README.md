# Mechanism recovery — a reproducible negative result (supplementary)

Two self-contained, CPU-only synthetic spikes testing a question the "Mechanisms Matter"
(Qi & Chapfuwa) line raises: does an **explicitly-estimated per-context causal influence matrix
`Â_C`** predict cross-context perturbation transportability **better than correlation baselines**?
Ground truth is known by construction, so the question is answerable cleanly. It is answered here,
and the answer is a well-characterized **negative** — which is the contribution.

---

## Three takeaways

**1. The result — `Â` does not beat correlation, in either regime tested.**
On the field's own simulator (CausalDGP), explicit per-context `Â_C` estimation fails to beat
correlation/additivity at predicting cross-context transportability, in **both** regimes we tested:
linear dynamics with single-gene held-out perturbations (spike #1) and nonlinear dynamics with
held-out double perturbations (spike #2). The pre-registered bar (mechanism must exceed *both*
correlation nulls with the gap CI excluding 0) is **not met → FAIL**, reported honestly.

| | Spike #1 (linear, singles) | Spike #2 (nonlinear, doubles) |
|---|---|---|
| best mechanism | AUROC **0.69** | mech−corr gap **≈ 0 across all λ** |
| best correlation null | AUROC **0.83** | (gap −0.003 → −0.001, CIs span 0) |
| oracle (true `A`) | **1.00** | linear-oracle 1.00 → 0.88 |
| verdict | FAIL | FAIL / PARK |

**2. Why — the positive core (this is the finding).**
In a linear-Gaussian OU system the stationary covariance solves the Lyapunov equation
`A Σ + Σ Aᵀ = −D`, so **`Σ` is a near-sufficient statistic for `A`**: correlation captures the
mechanism essentially for free, and *more stably* than a sparse interventional estimate under
`P≪G`. It even sharpens **faster with sequencing depth** than the mechanism (correlation
0.72→0.96 over 500→4000 cells; mechanism flat ~0.70). And **nonlinearity does not relax this**:
the estimator's ~20% `Â`-error swamps the ~18% epistasis signal it is meant to exploit — summing
*observed* singles predicts an epistatic double's effect at cosine ~0.98 while the mechanism reaches
only ~0.65. This is a concrete explanation for *why the field's correlation baselines are so hard to
beat, on the field's own simulator*.

**3. A standalone positive — the linear transportability condition degrades under nonlinearity.**
The linear/additive transportability condition (`τ = −A⁻¹Γ`, operating-point-blind) holds AUROC
**1.00** through moderate nonlinearity but falls to **0.88** under strong saturation (λ=0.85),
quantifying how far the field's linear assumption drifts from truth as biology becomes nonlinear.

---

## Spike #1 — linear regime, single-gene held-out perturbations

Linear-Gaussian CausalDGP (OU latent + NB emission), `G=50`, `P_train=30`, held-out singles perturbed
in *neither* context, modes {none, a, b, both}, 8 seeds. Held-out = never-perturbed genes, so the
mechanism must extrapolate.

| Method | Pooled AUROC [95% CI] |
|---|---|
| Oracle (true `A` columns) — *reference* | **1.000** |
| Null: correlation graph | **0.828** [0.757, 0.889] |
| Null: co-expression (GEARS) | 0.702 [0.647, 0.748] |
| **Mechanism — `Â` column-k (best variant)** | **0.689** [0.640, 0.732] |
| Mechanism — brief inversion predictor | 0.624 [0.549, 0.695] |

`gap(mechanism − corr_null) = −0.140 [−0.224, −0.053]` (CI excludes 0, wrong direction) → FAIL.
Artifacts: [`results/c4_auroc.csv`](results/c4_auroc.csv), [`results/moneyshot.png`](results/moneyshot.png),
[`results/sensitivity_ncells.png`](results/sensitivity_ncells.png) (depth sensitivity), full write-up in
[`FINDINGS.md`](FINDINGS.md).

## Spike #2 — nonlinear regime, held-out double perturbations

Nonlinear drift `h_λ(x)=(1−λ)x+λ·s·tanh(x/s)` (λ=0 recovers the linear system); test = held-out
doubles of *individually-perturbed* genes (fixes spike-1's coverage gap and introduces epistasis).
The hypothesis was that nonlinearity would dissolve correlation's two advantages. It does not.

| λ | epistasis (rel) | frac transportable | mechanism | corr-add | obs-add | linear oracle | gap (mech−corr) |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.50 | 0.997 | 1.000 | 1.000 | 1.000 | −0.003 |
| 0.50 | 0.06 | 0.50 | 0.988 | 0.998 | 1.000 | 1.000 | −0.010 |
| 0.85 | 0.18 | 0.30 | 0.982 | 0.983 | 0.999 | 0.876 | −0.001 |
| 1.00 | — | — | — | — | — | — | degenerate (dropped) |

The gap never crosses zero; the finer Spearman metric trends the *wrong* way (+0.02 → −0.03). The
manipulation is real (epistasis 0→0.18, transportable fraction 0.50→0.30). Artifacts:
[`results/gap_vs_lambda.png`](results/gap_vs_lambda.png), [`results/spike2_diagnostics.csv`](results/spike2_diagnostics.csv),
full write-up in [`FINDINGS_SPIKE2.md`](FINDINGS_SPIKE2.md).

---

## Reproduce (CPU-only, minutes; no GPU, no torch)

```bash
conda env create -f environment.yml    # or: pip install numpy scipy scikit-learn pandas matplotlib
python run_c4.py && python eval.py && python sensitivity.py     # spike #1  -> results/c4_auroc.csv, moneyshot.png, sensitivity_ncells.*
python run_spike2.py && python spike2_diag.py                   # spike #2  -> results/gap_vs_lambda.*, spike2_diagnostics.csv
```

Seeds are fixed; runs are deterministic and regenerate every committed artifact. `run_c4.py` prints the
**M0 simulator check** (empirical vs analytic effect, Pearson `r ≈ 0.96`) — if that fails, nothing
downstream is valid. Raw per-record CSVs (`*_records.csv`) are regenerated by the run commands and are
git-ignored; the curated artifacts in `results/` are committed.

### Files

| file | role |
|---|---|
| `causaldgp.py` | simulator: linear OU + NB emission (spike 1); nonlinear drift + fixed-point solver (spike 2) |
| `mechanism.py` | interventional estimator `estimate_A` + predictors + nulls; nonlinear double-predictor + additive nulls |
| `labels.py` | analytic transportability ground truth (Prop 1); nonlinear label + linear oracle |
| `run_c4.py` / `eval.py` / `sensitivity.py` | spike 1: M0 check + sweep, AUROC + bootstrap CIs, depth sensitivity |
| `run_spike2.py` / `spike2_diag.py` | spike 2: kill probe + λ grid, Spearman + effect-capture diagnostics |

### Documented in-source (methodological, not tuned on labels)

- **Lyapunov `−D` sign fix** — `solve_lyapunov(A, D)` returns a negative-definite covariance for Hurwitz
  `A`; the valid covariance needs `−D`.
- **Principled (non-inversion) predictor** — the naïve `solve(Â+ridge·I, −Γ)` sign-flips held-out effects
  (rank-deficient `Â`); replaced with the estimable causal-target column / identity-fill predictor.
- **`α = 0.002`, `n_cells = 1000`** — chosen on a *disjoint dev-seed set* by maximizing label-independent
  `A`-recovery (not any mechanism-vs-null gap).
- **Spike-2 `s = 0.4`** (not the illustrative 1.5) — a manipulation check showed 1.5 leaves the small
  operating point (`|x*|≈0.24`) in tanh's linear zone → ~0 epistasis → a *false* PARK; `s=0.4` yields a
  clean epistasis gradient while staying stable through λ=0.85.

---

## Scope & future work (honest)

Results are **synthetic-only**. The transportability signal is **real** (oracle with the true `A`
scores AUROC 1.0), but recovering it is **estimation-gated under `P≪G`**, and neither linearity nor
nonlinearity relaxes that gate. The one lever we did **not** test is a **materially better estimator**
— e.g. a joint sparse + low-rank `Â`, or estimating the effect map `B = −A⁻¹` directly from observed
effect columns rather than inverting a sparse `Â`. That is a research direction, not a hackathon task,
and it is out of scope here. Also out of scope by rule: real ZHU25 data, the A/B mechanism
decomposition, and any acquisition/decision layer.
