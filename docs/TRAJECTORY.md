# Phase TC — Trajectory-Coupling gate → **CLEAN NEGATIVE (do not build)**

*Overnight Dev-2 bet. Hypothesis: predictability is a **biological/dynamical** property, not a model
one — unrecoverable perturbations move the cell **along** the activation trajectory (a fixed axis)
rather than shifting an endpoint, so the identifiable target is a **scalar displacement** `s_p =
proj(δ_p, a)` along the activation axis, not the 3000-dim delta (proven per-perturbation noise-floored
at ~0.03). CP2 and the predictability-budget numbers are the frozen fallback and are untouched.*

**Provenance:** [IN-PROJECT] measured here on committed data (CZI pseudobulk + `budget_decomposition.csv`
+ frozen split fd2b8c21) · [INFERENCE] the hypothesis under test. First attempt against a measured
target, not a solved problem.

## Verdict — both gate parts FAIL → do not build. GPU not spent.

The CPU gate is two parts; **both must pass to authorize the build**; neither did.

### G-TC.0 — does the scalar target reproduce above the ~0.03 floor? → **NO (at floor)**

Activation axis `a` = Rest→Stim48hr non-targeting control-state shift (frozen normalization space).
Per-perturbation cross-donor reproducibility of `s_p` (Spearman across perturbations, donor-pair-averaged):

| context | `s` (raw proj) | `s_norm` (magnitude-normalized) | `s` on PC1 axis | ‖δ‖ magnitude (ref) | random-axis null |
|---|---|---|---|---|---|
| Rest | 0.079 | 0.082 | 0.061 | 0.215 | 0.028 |
| Stim8hr | 0.072 | 0.070 | 0.062 | 0.229 | 0.034 |
| Stim48hr | 0.068 | 0.064 | 0.065 | 0.248 | 0.029 |

The scalar reproduces at **~0.07** — barely above the random-axis null (**0.03**) and far below the
*magnitude* reproducibility (**0.21–0.25**, the trivially-reproducible quantity). The
magnitude-normalized projection (the non-trivial signal) is ~0.07 ≈ null. **Robust to the axis
choice** (control-shift vs PC1-of-states, cos 0.75: both ~0.06–0.08). Calibration is sound — the null
(0.03) matches the known delta floor and magnitude reproduces as expected, so this is a real result,
not a broken pipeline. **The scalar reframe does not escape the SNR wall.**

### G-TC.1 — does recoverability predict trajectory-coupling? (C-TC.1) → **NO (partial ρ ≈ 0)**

`TC_p = |proj(δ_p, a)|/‖δ_p‖`; `R_p` = committed do-operator `causal_frac_of_ceiling`. **Primary bar =
partial Spearman controlling for magnitude and reliability.**

| split | raw Spearman(R,TC) | **partial(\|mag,rel)** | partial (2D subspace) | perm p | per-donor sign |
|---|---|---|---|---|---|
| condition | +0.003 | **+0.007** | −0.028 | 0.754 | 2/4 |
| gene | +0.051 | **+0.034** | −0.006 | 0.549 | 3/4 |

Pre-registered PASS was |partial| ≥ 0.3, p < 0.01, sign ≥ 3/4. Observed partial ρ ≈ **0** (both 1D and
2D activation subspace), p ≫ 0.01. The raw Spearman is also ~0, so the partial isn't masking a real
signal — **recoverability and trajectory-coupling are simply unrelated** (see `figures/trajectory_coupling_gate.png`:
flat blobs). Additionally, `TC_p` is small for most perturbations (~0.05–0.20) — effects are largely
**orthogonal** to the activation axis to begin with.

## What this rules out (fold into the audit)

**Predictability is *not* a trajectory-geometry property.** The hard-to-recover perturbations are not
the ones that move the cell along the activation axis, and the proposed scalar target does not
reproduce above the noise floor regardless. This joins F1 (Â_C), F2 (fluctuation/3rd-moment), and the
CellCap SNR pre-check: the unification "the hard perturbations are dynamical" is **not supported** —
the per-perturbation frontier is noise-limited (SNR pre-check) *and* the residual is not
trajectory-axis-aligned (this gate). The remaining honest position stands: the per-perturbation
frontier is a **noise floor at this experimental depth**, not a modeling gap a reframed target unlocks.

## Fences respected (§F)

F1/F2 not re-entered. F3: every claim is fraction-of-ceiling / partial-correlation, not raw δ. **F4:
no trajectory ODE / vector field was fit — this gate only projects onto a *measured* axis and stops;
it makes no dynamical-identifiability claim.** No build ran.

## Deliverables & state

`results/trajectory_coupling_gate.csv`, `results/trajectory_coupling_perpert.csv`,
`figures/trajectory_coupling_gate.png`, `hypotheses.md` pre-registration (C-TC.1/C-TC.2). Committed to
branch `trajectory` (**unpushed, unmerged** — the lead reviews awake). **Box idle; no GPU spent.**
