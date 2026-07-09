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

**What the residual gene program IS  [VERIFIED — pseudobulk `gene_name` column].** The top residual
genes (what CIPHER's equilibrium Σu misses most) are **IFNG, IL2, IL3, IL13, CSF2 (GM-CSF),
CCL1/CCL3/CCL4, CXCL8, LTB, IL2RA**, plus TMSB4X/PFN1/S100A4/TRBC1. This is unambiguously the
**T-cell activation / effector-cytokine program** — the switch-like induction that turns on during
the Rest→Stim transition, peaks acutely at 8hr, and cannot be represented by a linear fluctuation-
response built on *resting* covariance. It is largely **shared** (perturbation-independent: cells
activate regardless of which gene is knocked down), which is exactly why B2 finds the *per-pert*
residual noise-limited while the *aggregate program* is 0.94-reproducible.

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

---

## B3 — routed build: route = **RED** (measure-and-stop)  [IN-PROJECT]

**Route decision.** Q3 fires (transition-peaked, 4/4 donors, effect/confound-controlled), confirmed
by B2 (the recoverable-at-pseudobulk part is a shared transient program; per-pert structure is noise).
Per the pre-registered routing, **Q3-fires → RED**: do **not** build a trajectory / non-autonomous
vector-field model. Three timepoints (Rest / Stim8hr / Stim48hr) cannot identify a driven vector
field; a long GPU run there would produce a *confident artifact*, which is worse than nothing.
**No build was started. GPU left idle** (per the brief: leave the box idle rather than manufacture
work when the route is RED). CP2/budget untouched; nothing merged.

**Robustness of the decisive fingerprint  [IN-PROJECT].** The transition-peak is clear in the
canonical CIPHER raw-count residual (`_cipher`: Rest 0.813 < Stim8hr 0.943 > Stim48hr 0.931, peak
+0.071). The `_exact` and log1p (`residual.csv`) variants are **degenerate** — residual saturates at
~0.995 (CIPHER explains ≈nothing; the log1p case is the normalization the sizer docstring flags as
broken), so they carry no fingerprint. Caveat: the fingerprint is a property of the *meaningful*
(raw-count, CIPHER-Methods) residual; my reconstruction matched it exactly (corr 1.000).

**What it is → what data it needs.**
- *What it is:* a **transient T-cell activation cytokine program** (IFNG, IL2, CSF2, IL3/IL13,
  CCL/CXCL chemokines, IL2RA…) induced across the Rest→Stim transition, peaking at 8hr — a
  far-from-equilibrium response outside the reach of an equilibrium fluctuation-response model.
  Largely *shared* across perturbations; the *perturbation-specific* residual is noise-limited at
  pseudobulk (cross-donor 0.03).
- *What it needs (for perturbation prediction, the project goal):* **single-cell resolution + more
  cells per perturbation** — to (a) lift the per-pert residual above the pseudobulk noise floor and
  (b) test whether it depends on within-condition cell-state (the CellCap seam). This is a raw-cell
  download, **flag-gated** (≈130 GB/stratum), **not** run unattended.
- *What it does NOT need:* more timepoints for *prediction* — the transient program is already
  condition-level-captured by per-condition models; denser timepoints would serve *mechanism*
  (identifying the transition vector field), which remains hard and is the RED-forbidden build.

**Pre-registered success bar (stated, not exercised).** Had a build run, it would "succeed" only by
recovering ≥25% of the do-operator-adjusted bucket-C remainder (≈46% of C), out-of-sample, beating
both Ridge and the do-operator. **No build ran → no claim.** This is a *reported route*, not a null
model result.

**Follow-on queue — honest assessment (no busywork).**
- *Item 1 (routed build):* skipped — RED.
- *Item 2 (do-operator-adjusted per-condition ablation):* the committed do-operator predictions are
  aggregated by hold-out split (gene / condition), not resolvable by the 3 biological conditions, so a
  clean per-condition do-operator ablation is **not computable from committed data** (needs per-
  condition eval predictions that were not committed).
- *Item 3 (second pseudobulk tool):* would not beat the pseudobulk noise floor (B2) → no information
  gain; not run.
- *Item 4 (GREEN sweep):* N/A (GREEN did not run).
→ Per the brief, remaining queue items are exhausted/uninformative on committed data → **STOP, GPU
idle.** The one high-value next step (cell download → CellCap test) is flag-gated and recommended to
the lead below.

---

## State for the lead + recommended next step

**Bottom line.** The bucket-C residual is a **transient T-cell activation cytokine program** — real,
structured, reproducible (B1), and it PEAKS at the Rest→Stim8hr activation transition. But its
*reproducible* part is a **shared, perturbation-independent** program (captured by any per-condition
model); the **perturbation-specific** residual — the actual prediction frontier — is **noise-limited
at the pseudobulk level** (cross-donor 0.03). The existing state-aware tool (do-operator) already
leaves ~46% of gene bucket C. **Route = RED: no static/trajectory model built on the current
pseudobulk data would honestly clear the bar; the GPU was left idle rather than fill the night with a
doomed run.**

**Recommended next step (FLAG-GATED — needs your authorization).** Download 1–2 Stim8hr strata of raw
cells (~130 GB each, same-region public S3 `--no-sign-request`, delete-after — the established C-NL
pattern) to run the **decisive test the pseudobulk cannot**: (1) does per-perturbation residual
structure appear above noise at single-cell SNR, and (2) does it depend on within-condition cell-state
(the **CellCap** seam)? If yes → a GREEN CellCap build becomes justified (no extra timepoints). If no
→ the residual is genuinely a shared transient program and the honest headline is "predictability
ceiling on this axis; the per-pert frontier needs a different experiment (denser transition sampling)."
I did **not** run this unattended (>5 GB download flag + RED routing).

**What I did / did not do.** Did: B1 localization + B2 recovery, committed (`3ef19ca`, `6ea0e65`).
Did not: build any model, download cells, touch CP2 / the budget / Dev 4's branches, or merge.

**Box line at hand-off.** GPU **idle** (0% util); no Phase-B job running or queued; the box is free.
Committed artifacts on branch `phaseB` (unpushed, unmerged). Scratch on box at `~/cd4-phaseB/`.

---

## Step 0 — SNR pre-check for the single-cell CellCap seam → **NOT GREEN (confirmed floor)**  [IN-PROJECT]

The lead greenlit the ~130 GB single-cell run *conditional on* an SNR pre-check: is the ~0.033
per-perturbation reproducibility a **pseudobulk-aggregation artifact single-cell resolution fixes**,
or a **genuine effect-size/noise floor** it cannot? Computed from committed data only (no download).

**Decisive fact:** single-cell modeling of a perturbation's *mean* effect does **not** add cells —
pseudobulk is the sufficient statistic for the mean, so the same ~180 cells/pert set the floor either
way. Single-cell helps only via within-state **concentration** (effect localized to a subpopulation)
or higher moments.

| metric | value | reading |
|---|---|---|
| cross-donor reproducibility, full 3000 genes | 0.029 | matches Phase B's ~0.033 |
| cross-donor reproducibility, **relevant top-50 effect genes** | **0.033** | focusing on the genes that matter does **not** lift it |
| perturbations with detectable effect (SNR>3) | **17%** | most per-pert effects are weak at 180 cells |
| within-donor reliability (noise model) | 0.48 | but ≫ cross-donor 0.03 → the non-noise structure is **donor-specific**, not reproducible biology single-cell repairs |
| cells needed to reach usable r=0.30 | **~12×** | not available (experimental cell count is fixed) |
| concentration needed instead | **~8% of cells** | stringent; even a generous f=0.3 projects to only r≈0.10 |

**The single number for the lead: per-perturbation cross-donor reproducibility on the relevant genes
= 0.033 (unchanged from the full-vector floor); projected single-cell best-case ≈ 0.10, short of the
0.30 usable floor.** → **NOT GREEN.** The per-perturbation frontier is a genuine effect-size/noise (and
donor-variability) floor, **not** a pseudobulk artifact single-cell resolution fixes. **No download,
no build; the 130 GB is not spent.** Box stays down.

*Honest caveat (the one scenario single-cell might help):* the within-donor 0.48 vs cross-donor 0.03
gap *could* be donor differences in cell-state composition (which cell-state-conditioning would
disentangle) rather than pure batch. The pre-check can't separate these without cells — but the
underlying effects are weak (17% detectable, median SNR 1.7), so even in that best case the expected
recovery is low. Recommendation stands: **do not spend the download on an ~8%-concentration bet over
weak effects.** If the lead wants to chase the composition hypothesis specifically, that is the only
justification — a deliberate low-probability bet, not the default.

Outputs: `results/phaseB_snr_precheck.csv`, `phaseB_snr_precheck_summary.csv`.

**Result type:** a clean, reported **confirmed floor** ("the per-perturbation frontier is noise-limited
even at single-cell resolution"), exactly the pre-registered NOT-GREEN outcome — no GPU-hours, no
egress spent to reach it.

