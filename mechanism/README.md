# Mechanism & fluctuation-response probes — two negatives, one positive (supplementary)

Three self-contained, CPU-only synthetic probes on the "Mechanisms Matter" (Qi & Chapfuwa) /
CIPHER line, each proven on ground truth where the answer is known. The arc:

1. **Probe #1 (linear, single held-out perturbations)** — does an explicitly-estimated per-context
   causal matrix `Â_C` beat correlation baselines at cross-context transportability? → **FAIL**.
2. **Probe #2 (nonlinear, held-out doubles)** — does nonlinearity/epistasis let `Â_C` overtake
   correlation? → **FAIL / PARK**.
3. **Probe #3, the C-NL gate (third moment)** — does the *baseline third moment* predict the
   perturbation response that covariance provably cannot? → **LIVE** (the positive).

The through-line: in a linear-Gaussian system the control covariance solves the Lyapunov equation, so
`Σ` is a near-sufficient statistic for the mechanism — which is *why* correlation baselines are
unbeatable (#1–#2). The one thing covariance cannot carry is the non-Gaussian (third-moment) structure
of the fluctuations — and that is exactly where #3 found real, estimable signal.

---

## Probe #1 — mechanism recovery, linear regime → FAIL

`G=50`, `P_train=30`, held-out singles, 4 modes, 8 seeds.

| Method | Pooled AUROC [95% CI] |
|---|---|
| Oracle (true `A` columns) | **1.000** |
| Null: correlation graph | **0.828** [0.757, 0.889] |
| Null: co-expression (GEARS) | 0.702 [0.647, 0.748] |
| **Mechanism (column-k, best)** | **0.689** [0.640, 0.732] |

`gap(mech − corr) = −0.140 [−0.224, −0.053]` → FAIL. Oracle=1.0 ⇒ signal real, `Â`-estimation under
P≪G is the wall; correlation sharpens *faster* with depth (0.72→0.96). Details: [`FINDINGS.md`](FINDINGS.md),
`results/{c4_auroc.csv, moneyshot.png, sensitivity_ncells.png}`.

## Probe #2 — mechanism recovery, nonlinear regime (doubles) → FAIL / PARK

`h_λ(x)=(1−λ)x+λ·s·tanh(x/s)`, held-out doubles of individually-perturbed genes.

| λ | epistasis | mechanism | corr-add | obs-add | linear oracle | gap |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.997 | 1.000 | 1.000 | 1.000 | −0.003 |
| 0.85 | 0.18 | 0.982 | 0.983 | 0.999 | 0.876 | −0.001 |

Gap stays flat across λ; summing *observed* singles predicts the epistatic double at cos 0.98 while the
mechanism reaches 0.65. Standalone positive: the linear transportability condition degrades AUROC
1.00→0.88 under nonlinearity. Details: [`FINDINGS_SPIKE2.md`](FINDINGS_SPIKE2.md),
`results/{gap_vs_lambda.png, spike2_diagnostics.csv}`.

## Probe #3 — the C-NL gate (third moment) → LIVE

Symmetric (equilibrium) `A` so CIPHER's `Σu` is the exact first-order response (σ²=2 ⇒ `Σ=−A⁻¹`,
clean λ=0), genuine nonlinear-SDE Euler–Maruyama sampling (needed for a nonzero third moment), nonzero
baseline `b` (else the odd `tanh` gives a symmetric law with zero third moment). Pure second-order
response `c_ik = [Δμ_i(+m e_k)+Δμ_i(−m e_k)]/(2m²)`.

| λ | R²_T | R²_cov | ΔR² [95% CI] |
|---|---|---|---|
| 0.00 | 0.00 | 0.00 | −0.000 [−0.002, +0.000] |
| 0.50 | 0.73 | 0.00 | **+0.733** [+0.673, +0.772] |
| 0.85 | 0.76 | 0.01 | **+0.749** [+0.640, +0.802] |

The baseline third moment `T_ikk` explains ~61–76% of the second-order response; the covariance
surrogate `Σ_ik²` explains ~0%. **Test 3**: survives to 1,000 control cells (latent ΔR² 0.75→0.61;
NB-emission-on 0.25→0.19 — emission costs ~2/3 of the signal but holds). **Honest caveats**: the
nonlinear term is *small* (~3–4% of the response — prediction is strong, magnitude on real data is a
separate sizing question); the gate is equilibrium-only (necessary-not-sufficient for real non-equilibrium
networks); it is a *response* predictor, **not** causation (does not resurrect `Â_C`). Details:
[`FINDINGS_CNL.md`](FINDINGS_CNL.md), `results/{delta_r2_vs_lambda.png, depth_threshold.png}`.

**Provenance discipline** (kept throughout `FINDINGS_CNL.md`): the third-moment link is an *inference*
from fluctuation-response theory, **never** attributed to CIPHER; the coefficient is *fit*, not assumed ½.

---

## Reproduce (CPU-only, no GPU, no torch)

```bash
conda env create -f environment.yml
python run_c4.py && python eval.py && python sensitivity.py     # probe #1
python run_spike2.py && python spike2_diag.py                   # probe #2
python run_cnl_gate.py                                          # probe #3 (C-NL); `quick` for a fast read
```

Seeds fixed; deterministic; regenerates every committed artifact. Raw per-record CSVs are git-ignored;
curated `results/` artifacts are committed.

### Files

| file | role |
|---|---|
| `causaldgp.py` | simulator: linear OU + NB emission (#1); nonlinear drift + fixed points (#2); symmetric `A` + nonlinear-SDE Euler–Maruyama sampler (#3) |
| `mechanism.py` / `labels.py` | estimators, nulls, and analytic labels for #1–#2 |
| `response.py` | #3: third moment `T`, `second_order_term`, covariance surrogate, ΔR² |
| `run_c4.py` / `eval.py` / `sensitivity.py` | probe #1 |
| `run_spike2.py` / `spike2_diag.py` | probe #2 |
| `run_cnl_gate.py` | probe #3 (Tests 1–3) |

## Scope guard

Out of scope by rule (all probes): real ZHU25 data, the A/B decomposition, acquisition, and — for #3 —
the real-data build (closed-form `Σu + c·T[u,u]`, low-rank `T`) and the analytic-½ derivation. Those are
gated on these results and are the project lead's call.
