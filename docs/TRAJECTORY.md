# Phase TC — Trajectory-Coupling gate → **CLEAN NEGATIVE (do not build)**

*Overnight Dev-2 investigation. The hypothesis under test is that predictability is a biological or
dynamical property rather than a modelling one: unrecoverable perturbations move the cell along the
activation trajectory (a fixed axis) rather than shifting an endpoint, so the identifiable target is a
scalar displacement `s_p = proj(δ_p, a)` along the activation axis, rather than the 3000-dim delta
(established as per-perturbation noise-floored at approximately 0.03). CP2 and the predictability-budget
numbers constitute the frozen fallback and remain untouched.*

**Provenance:** [IN-PROJECT] measured here on committed data (CZI pseudobulk + `budget_decomposition.csv`
+ frozen split fd2b8c21) · [INFERENCE] the hypothesis under test. This is a first attempt against a measured
target, not a solved problem.

## Verdict — both gate parts FAIL → do not build. GPU not spent.

The CPU gate comprises two parts; both must pass to authorise the build, and neither did.

### G-TC.0 — does the scalar target reproduce above the ~0.03 floor? → NO (at floor)

The activation axis `a` is the Rest→Stim48hr non-targeting control-state shift (frozen normalization space).
Per-perturbation cross-donor reproducibility of `s_p` (Spearman across perturbations, donor-pair-averaged)
is reported below:

| context | `s` (raw proj) | `s_norm` (magnitude-normalized) | `s` on PC1 axis | ‖δ‖ magnitude (ref) | random-axis null |
|---|---|---|---|---|---|
| Rest | 0.079 | 0.082 | 0.061 | 0.215 | 0.028 |
| Stim8hr | 0.072 | 0.070 | 0.062 | 0.229 | 0.034 |
| Stim48hr | 0.068 | 0.064 | 0.065 | 0.248 | 0.029 |

The scalar reproduces at approximately 0.07, only marginally above the random-axis null of 0.03 and far
below the magnitude reproducibility of 0.21–0.25 (the trivially reproducible quantity). The
magnitude-normalized projection, which carries the non-trivial signal, is approximately 0.07, effectively
equal to the null. The result is robust to the choice of axis (control-shift versus PC1-of-states, cos
0.75: both approximately 0.06–0.08). Calibration is sound: the null (0.03) matches the known delta floor
and the magnitude reproduces as expected, so this is a genuine result rather than a broken pipeline. The
scalar reframe therefore does not escape the SNR wall.

### G-TC.1 — does recoverability predict trajectory-coupling? (C-TC.1) → NO (partial ρ ≈ 0)

Here `TC_p = |proj(δ_p, a)|/‖δ_p‖` and `R_p` is the committed do-operator `causal_frac_of_ceiling`. The
primary criterion is the partial Spearman correlation controlling for magnitude and reliability.

| split | raw Spearman(R,TC) | **partial(\|mag,rel)** | partial (2D subspace) | perm p | per-donor sign |
|---|---|---|---|---|---|
| condition | +0.003 | **+0.007** | −0.028 | 0.754 | 2/4 |
| gene | +0.051 | **+0.034** | −0.006 | 0.549 | 3/4 |

The pre-registered PASS threshold was |partial| ≥ 0.3, p < 0.01, sign ≥ 3/4. The observed partial ρ is
approximately 0 (in both the 1D and the 2D activation subspace), with p ≫ 0.01. The raw Spearman is also
approximately 0, so the partial correlation is not masking a real signal: recoverability and
trajectory-coupling are simply unrelated (see `figures/trajectory_coupling_gate.png`, which shows flat
blobs). In addition, `TC_p` is small for most perturbations (approximately 0.05–0.20), indicating that
effects are largely orthogonal to the activation axis to begin with.

## What this rules out (fold into the audit)

Predictability is not a trajectory-geometry property. The hard-to-recover perturbations are not the ones
that move the cell along the activation axis, and the proposed scalar target does not reproduce above the
noise floor in any case. This joins F1 (Â_C), F2 (fluctuation/3rd-moment), and the CellCap SNR pre-check:
the unification that "the hard perturbations are dynamical" is not supported. The per-perturbation frontier
is noise-limited (SNR pre-check), and the residual is not trajectory-axis-aligned (this gate). The
remaining honest position stands: the per-perturbation frontier is a noise floor at this experimental
depth, not a modelling gap that a reframed target unlocks.

## Fences respected (§F)

F1 and F2 were not re-entered. F3: every claim is expressed as fraction-of-ceiling or partial correlation,
not raw δ. F4: no trajectory ODE or vector field was fit; this gate only projects onto a measured axis and
stops, and it makes no dynamical-identifiability claim. No build ran.

## Deliverables & state

`results/trajectory_coupling_gate.csv`, `results/trajectory_coupling_perpert.csv`,
`figures/trajectory_coupling_gate.png`, and `hypotheses.md` pre-registration (C-TC.1/C-TC.2), committed to
branch `trajectory` (unpushed and unmerged, pending the lead's review). The box is idle and no GPU was spent.
