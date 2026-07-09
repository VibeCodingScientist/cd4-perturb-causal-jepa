# Phase B — Localize the residual, test existing machinery, route the build

*Overnight autonomous session (Dev 2). Characterizes the structured, reproducible, unrecovered
"bucket C" residual that the Phase-A predictability budget surfaced on gene generalization.
CP2 and the committed budget numbers are the frozen fallback and are **not** touched here.*

**Provenance tiers.** `[IN-PROJECT]` measured here on committed data · `[VERIFIED]` confirmed
against a primary · `[INFERENCE]` reasoning to be tested. Framed as a first attempt against a
measured target, not a solved problem.

> **⚠️ Honesty seam — two different "residuals" (read before interpreting).** Phase A's *bucket C*
> is what a **fitted linear map (Ridge)** misses on the **eval metric** (top-50-DEG Pearson-δ, log
> pseudobulk space); on the gene split it is ≈0.76 of the reliability ceiling, of which the
> do-operator already recovers ≈56%. The object localized in **B1** is Dev 4's committed **C-NL
> first-order residual** `r = ΔX − Σu` — CIPHER's **single-gene** linear-response residual in
> **raw-count** space (median resid_frac ≈0.9, R²≈0.15). These are related but **not identical**:
> the CIPHER residual is a *looser, larger* object (a single-gene response leaves more than a full
> linear map). So B1's residual **magnitude** overstates bucket C; what transfers between them is
> the residual's **structure** — where the unexplained response concentrates. B1's routing signal
> (Q3, the transition fingerprint) is a statement about that structure and is robust to the space
> difference. B2 bridges the two explicitly.

---

## B1 — localization of the C-NL residual `r = ΔX − Σu`  [IN-PROJECT]

Reconstructed the residual from the 12 committed per-stratum checkpoints
(`cnl_ckpt_donor_*_<cond>.npz`: Σ = raw-count control covariance, ΔX = per-perturbation raw mean
shift, 3000 HVG). **Reconstruction self-check: corr = 1.000, MAD = 0.0000 vs the committed
`cnl_realdata_residual_cipher.csv` across all 12 strata** — the gene-level decomposition is exact.

| Q | Question | Result | Fires? |
|---|---|---|---|
| **Q1** | Which genes? (STOP gate) | Top-50/3000 genes carry **25.8%** of residual mass; gene-residual profile corr **0.941 within-condition** (across donors) vs 0.417 cross-condition → a reproducible, **condition-specific gene program** | **STRUCTURED → PROCEED** |
| **Q2** | Which perturbations? | Cross-donor per-pert reproducibility low (0.03–0.09), CV 0.14–0.26 → residual is broad across perturbations, **not a specific pert class** | no |
| **Q3** | **Which condition? (DECISIVE)** | Rest 0.81 < **Stim8hr 0.94** > Stim48hr 0.93; transition-peak `Stim8hr − mean(Rest,Stim48hr)` = **+0.064, 4/4 donors positive**; effect-controlled OLS Stim8hr **+0.185**, Stim48hr +0.122 vs Rest (β_effect −0.116, confound runs backwards); n_ctrl confound ruled out (corr −0.002) | **FIRES → RED (transient)** |
| **Q4** | Cell-state dependent? | Within-pert residual SD across states = 0.117 (state-dependent), but of the **peaked/transient** kind per Q3. Within-condition cell-level test `[needs-cells]` | partial (transient) |
| **Q5** | Mean vs distribution? | Residual concentrates in high-variance genes (corr 0.71 with control variance). Full 2nd-moment (perturbed-cell covariance) `[needs-cells]`; 3rd moment already orthogonal 12/12 — **not re-tested** | proxy only |

**B1 verdict — Q1 STRUCTURED (passes STOP gate) + Q3 FIRES → the residual is a reproducible,
condition-specific gene program that PEAKS at the Rest→Stim activation transition (Stim8hr) and
partially relaxes by 48hr.** This is the **transient / far-from-equilibrium fingerprint**: CIPHER's
equilibrium fluctuation-response holds at Rest (R²≈0.34–0.62) and breaks during active activation
(R²≈0.11–0.14). `[INFERENCE → tested in B2]`

**Interpretation of the checkerboard (Q1 heatmap):** the *same genes* carry the residual across all
4 donors within a state (biological, not batch), but the *program differs by state* — consistent
with a state-specific transient, not a single static program.

Outputs: `results/phaseB_localization.csv`, `figures/phaseB_localization.png`.

**Route lean after B1: RED.** B2 tests whether an existing state-aware tool nonetheless recovers the
residual (which would overturn RED); B3 executes the confirmed route.

---

## B2 — recovery baseline: does an existing state-aware tool close the gap?  [IN-PROJECT]

**(A) The do-operator-adjusted remainder (the true denominator).** From the committed Phase-A budget
(gene split): bucket C ≈ 0.76 of the reliability ceiling; the committed do-operator (causal) recovers
**54% of C**, leaving a **do-operator-adjusted remainder ≈ 46% of C (0.35 of total)**. *Any* build must
beat that remainder, not all of C. (On the *condition* split the do-operator is beaten by linear Ridge,
so bucket C there is a gene-split phenomenon — the relevant axis.)

**(B) Is the remainder recoverable structure, or noise/shared? (the RED-vs-GREEN arbiter.)**
Reconstructed per-perturbation residual vectors from the committed checkpoints and measured cross-donor
reproducibility:

| level | cross-donor reproducibility |
|---|---|
| aggregate gene-residual profile (B1, averaged over ~1300 perts) | **0.94** (within-condition) |
| **individual perturbation** residual vector | **0.034** |
| individual perturbation residual, **shared program removed** | **0.033** |

**Reading.** The reproducible residual is a **shared, condition-specific gene program** (systematic
across all perturbations — hence 0.94 in the average) that peaks at the Stim8hr transition. The
**perturbation-specific** residual is **near noise at pseudobulk** (0.03; unchanged when the shared
program is removed → ~97% of a single perturbation's residual is sampling noise at ~180 cells/pert).
So no *pseudobulk* tool — CIPHER, Ridge, or the do-operator — can recover per-perturbation residual
structure that isn't resolvable at this SNR. **The gap is real (46% of C) but its per-perturbation,
recoverable part is obscured by sampling noise at the pseudobulk level.**

**B2 verdict → RED confirmed, and the data-need sharpened.** The residual is (i) a shared *transient*
activation program (condition-level, already captured by any per-condition model — not the prediction
frontier) plus (ii) a perturbation-specific part that is noise-limited at pseudobulk. Recovering (ii)
needs **single-cell resolution + more cells per perturbation** (to beat the noise floor and test
within-condition cell-state dependence, the CellCap seam) — i.e. a raw-cell download, **not** a
cleverer pseudobulk model. Output: `results/phaseB_recovery.csv`.

