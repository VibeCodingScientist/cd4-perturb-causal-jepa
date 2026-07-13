# Mechanism & fluctuation-response probes — two negatives, one positive (supplementary)

Three self-contained, CPU-only synthetic probes on the "Mechanisms Matter" (Qi & Chapfuwa) /
CIPHER line are presented here, each validated on ground truth where the answer is known. The
sequence is as follows.

1. **Probe #1 (linear, single held-out perturbations)** examines whether an explicitly-estimated
   per-context causal matrix `Â_C` outperforms correlation baselines at cross-context
   transportability. The verdict is FAIL.
2. **Probe #2 (nonlinear, held-out doubles)** examines whether nonlinearity and epistasis allow
   `Â_C` to overtake correlation. The verdict is FAIL / PARK.
3. **Probe #3, the C-NL gate (third moment)** examines whether the *baseline third moment* predicts
   the perturbation response that covariance provably cannot. This probe is LIVE and constitutes the
   positive result.

The unifying argument is that, in a linear-Gaussian system, the control covariance solves the
Lyapunov equation, so `Σ` is a near-sufficient statistic for the mechanism. This is precisely why the
correlation baselines are difficult to beat in Probes #1 and #2. The one quantity covariance cannot
carry is the non-Gaussian (third-moment) structure of the fluctuations, and that is exactly where
Probe #3 identified real, estimable signal.

---

## Probe #1 — mechanism recovery, linear regime → FAIL

`G=50`, `P_train=30`, held-out singles, 4 modes, 8 seeds.

| Method | Pooled AUROC [95% CI] |
|---|---|
| Oracle (true `A` columns) | **1.000** |
| Null: correlation graph | **0.828** [0.757, 0.889] |
| Null: co-expression (GEARS) | 0.702 [0.647, 0.748] |
| **Mechanism (column-k, best)** | **0.689** [0.640, 0.732] |

The result `gap(mech − corr) = −0.140 [−0.224, −0.053]` yields a verdict of FAIL. Because Oracle=1.0,
the signal is real; the limiting factor is `Â`-estimation under P≪G. Correlation sharpens *faster*
with depth (0.72→0.96). Further details are given in [`FINDINGS.md`](FINDINGS.md) and
`results/{c4_auroc.csv, moneyshot.png, sensitivity_ncells.png}`.

## Probe #2 — mechanism recovery, nonlinear regime (doubles) → FAIL / PARK

`h_λ(x)=(1−λ)x+λ·s·tanh(x/s)`, held-out doubles of individually-perturbed genes.

| λ | epistasis | mechanism | corr-add | obs-add | linear oracle | gap |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.997 | 1.000 | 1.000 | 1.000 | −0.003 |
| 0.85 | 0.18 | 0.982 | 0.983 | 0.999 | 0.876 | −0.001 |

The gap remains flat across λ; summing *observed* singles predicts the epistatic double at cos 0.98,
whereas the mechanism reaches 0.65. A standalone positive result is that the linear transportability
condition degrades AUROC from 1.00 to 0.88 under nonlinearity. Further details are given in
[`FINDINGS_SPIKE2.md`](FINDINGS_SPIKE2.md) and `results/{gap_vs_lambda.png, spike2_diagnostics.csv}`.

## Probe #3 — the C-NL gate (third moment) → LIVE

The construction uses a symmetric (equilibrium) `A` so that CIPHER's `Σu` is the exact first-order
response (σ²=2 ⇒ `Σ=−A⁻¹`, clean λ=0), genuine nonlinear-SDE Euler–Maruyama sampling (required for a
nonzero third moment), and a nonzero baseline `b` (otherwise the odd `tanh` gives a symmetric law with
zero third moment). The quantity of interest is the pure second-order response
`c_ik = [Δμ_i(+m e_k)+Δμ_i(−m e_k)]/(2m²)`.

| λ | R²_T | R²_cov | ΔR² [95% CI] |
|---|---|---|---|
| 0.00 | 0.00 | 0.00 | −0.000 [−0.002, +0.000] |
| 0.50 | 0.73 | 0.00 | **+0.733** [+0.673, +0.772] |
| 0.85 | 0.76 | 0.01 | **+0.749** [+0.640, +0.802] |

The baseline third moment `T_ikk` explains approximately 61–76% of the second-order response, while
the covariance surrogate `Σ_ik²` explains approximately 0%. **Test 3** confirms that the effect
survives to 1,000 control cells (latent ΔR² 0.75→0.61; NB-emission-on 0.25→0.19 — emission costs
approximately 2/3 of the signal but the effect holds). Several honest caveats apply. The nonlinear
term is *small* (approximately 3–4% of the response — the prediction is strong, but its magnitude on
real data is a separate sizing question); the gate is equilibrium-only (necessary but not sufficient
for real non-equilibrium networks); and it is a *response* predictor, not causation (it does not
resurrect `Â_C`). Further details are given in [`FINDINGS_CNL.md`](FINDINGS_CNL.md) and
`results/{delta_r2_vs_lambda.png, depth_threshold.png}`.

**Provenance discipline** (maintained throughout `FINDINGS_CNL.md`): the third-moment link is an
*inference* from fluctuation-response theory and is never attributed to CIPHER; the coefficient is
*fit* rather than assumed to be ½.

---

## Reproduce (CPU-only, no GPU, no torch)

```bash
conda env create -f environment.yml
python run_c4.py && python eval.py && python sensitivity.py     # probe #1
python run_spike2.py && python spike2_diag.py                   # probe #2
python run_cnl_gate.py                                          # probe #3 (C-NL); `quick` for a fast read
```

Seeds are fixed and the runs are deterministic, regenerating every committed artifact. Raw per-record
CSVs are git-ignored; curated `results/` artifacts are committed.

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

The following are out of scope by rule (for all probes): real ZHU25 data, the A/B decomposition,
acquisition, and — for #3 — the real-data build (closed-form `Σu + c·T[u,u]`, low-rank `T`) and the
analytic-½ derivation. These are gated on the present results and remain the project lead's decision.
