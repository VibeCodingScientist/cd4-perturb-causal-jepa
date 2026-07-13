# Phase B — Localising the residual, testing existing machinery, and routing the build

*This document reports an overnight autonomous session (Dev 2). It characterises the structured,
reproducible, and unrecovered "bucket C" residual that the Phase-A predictability budget surfaced on
gene generalisation. CP2 and the committed budget numbers constitute the frozen fallback and are **not**
modified here.*

**Provenance tiers.** `[IN-PROJECT]` denotes results measured here on committed data; `[VERIFIED]`
denotes results confirmed against a primary source; `[INFERENCE]` denotes reasoning that remains to be
tested. The work is framed as a first attempt against a measured target rather than a solved problem.

> **⚠️ Honesty seam — two different "residuals" (please read before interpreting).** Phase A's *bucket
> C* is the quantity that a **fitted linear map (Ridge)** misses on the **evaluation metric** (top-50-DEG
> Pearson-δ, in log pseudobulk space); on the gene split it is approximately 0.76 of the reliability
> ceiling, of which the do-operator already recovers approximately 56%. The object localised in **B1**
> is Dev 4's committed **C-NL first-order residual** `r = ΔX − Σu` — CIPHER's **single-gene**
> linear-response residual in **raw-count** space (median resid_frac ≈0.9, R²≈0.15). These quantities
> are related but **not identical**: the CIPHER residual is a looser and larger object, because a
> single-gene response leaves more unexplained than a full linear map. Consequently, the residual
> **magnitude** in B1 overstates bucket C; what transfers between the two is the residual's
> **structure** — that is, where the unexplained response concentrates. B1's routing signal (Q3, the
> transition fingerprint) is a statement about that structure and is robust to the difference in space.
> B2 bridges the two explicitly.

---

## B1 — localisation of the C-NL residual `r = ΔX − Σu`  [IN-PROJECT]

The residual was reconstructed from the 12 committed per-stratum checkpoints
(`cnl_ckpt_donor_*_<cond>.npz`: Σ = raw-count control covariance, ΔX = per-perturbation raw mean shift,
3000 HVG). **The reconstruction self-check yields corr = 1.000 and MAD = 0.0000 against the committed
`cnl_realdata_residual_cipher.csv` across all 12 strata**, indicating that the gene-level decomposition
is exact.

| Q | Question | Result | Fires? |
|---|---|---|---|
| **Q1** | Which genes? (STOP gate) | Top-50/3000 genes carry **25.8%** of residual mass; gene-residual profile corr **0.941 within-condition** (across donors) vs 0.417 cross-condition → a reproducible, **condition-specific gene program** | **STRUCTURED → PROCEED** |
| **Q2** | Which perturbations? | Cross-donor per-pert reproducibility low (0.03–0.09), CV 0.14–0.26 → residual is broad across perturbations, **not a specific pert class** | no |
| **Q3** | **Which condition? (DECISIVE)** | Rest 0.81 < **Stim8hr 0.94** > Stim48hr 0.93; transition-peak `Stim8hr − mean(Rest,Stim48hr)` = **+0.064, 4/4 donors positive**; effect-controlled OLS Stim8hr **+0.185**, Stim48hr +0.122 vs Rest (β_effect −0.116, confound runs backwards); n_ctrl confound ruled out (corr −0.002) | **FIRES → RED (transient)** |
| **Q4** | Cell-state dependent? | Within-pert residual SD across states = 0.117 (state-dependent), but of the **peaked/transient** kind per Q3. Within-condition cell-level test `[needs-cells]` | partial (transient) |
| **Q5** | Mean vs distribution? | Residual concentrates in high-variance genes (corr 0.71 with control variance). Full 2nd-moment (perturbed-cell covariance) `[needs-cells]`; 3rd moment already orthogonal 12/12 — **not re-tested** | proxy only |

**B1 verdict — Q1 STRUCTURED (passes the STOP gate) and Q3 FIRES: the residual is a reproducible,
condition-specific gene program that peaks at the Rest→Stim activation transition (Stim8hr) and
partially relaxes by 48hr.** This constitutes the transient, far-from-equilibrium fingerprint: CIPHER's
equilibrium fluctuation-response holds at Rest (R²≈0.34–0.62) and breaks down during active activation
(R²≈0.11–0.14). `[INFERENCE → tested in B2]`

**Interpretation of the checkerboard (Q1 heatmap).** The *same genes* carry the residual across all 4
donors within a state (a biological rather than a batch effect), whereas the *program differs by state*.
This pattern is consistent with a state-specific transient rather than a single static program.

**Identity of the residual gene program  [VERIFIED — pseudobulk `gene_name` column].** The top residual
genes — those that CIPHER's equilibrium Σu misses most — are **IFNG, IL2, IL3, IL13, CSF2 (GM-CSF),
CCL1/CCL3/CCL4, CXCL8, LTB, IL2RA**, together with TMSB4X/PFN1/S100A4/TRBC1. This is unambiguously the
**T-cell activation / effector-cytokine program**: the switch-like induction that turns on during the
Rest→Stim transition, peaks acutely at 8hr, and cannot be represented by a linear fluctuation-response
built on *resting* covariance. The program is largely **shared** (perturbation-independent, since cells
activate regardless of which gene is knocked down), which is precisely why B2 finds the *per-pert*
residual noise-limited while the *aggregate program* is 0.94-reproducible.

Outputs: `results/phaseB_localization.csv`, `figures/phaseB_localization.png`.

**Route lean after B1: RED.** B2 tests whether an existing state-aware tool nonetheless recovers the
residual, which would overturn RED; B3 executes the confirmed route.

---

## B2 — recovery baseline: does an existing state-aware tool close the gap?  [IN-PROJECT]

**(A) The do-operator-adjusted remainder (the true denominator).** From the committed Phase-A budget
(gene split): bucket C ≈ 0.76 of the reliability ceiling; the committed do-operator (causal) recovers
**54% of C**, leaving a **do-operator-adjusted remainder ≈ 46% of C (0.35 of total)**. *Any* build must
beat that remainder rather than all of C. On the *condition* split the do-operator is beaten by linear
Ridge, so bucket C there is a gene-split phenomenon — the relevant axis.

**(B) Is the remainder recoverable structure, or noise/shared? (the RED-vs-GREEN arbiter.)**
Per-perturbation residual vectors were reconstructed from the committed checkpoints, and cross-donor
reproducibility was measured:

| level | cross-donor reproducibility |
|---|---|
| aggregate gene-residual profile (B1, averaged over ~1300 perts) | **0.94** (within-condition) |
| **individual perturbation** residual vector | **0.034** |
| individual perturbation residual, **shared program removed** | **0.033** |

**Reading.** The reproducible residual is a **shared, condition-specific gene program** (systematic
across all perturbations, hence 0.94 in the average) that peaks at the Stim8hr transition. The
**perturbation-specific** residual is **near noise at pseudobulk** (0.03; unchanged when the shared
program is removed, implying that approximately 97% of a single perturbation's residual is sampling
noise at approximately 180 cells/pert). No *pseudobulk* tool — CIPHER, Ridge, or the do-operator — can
therefore recover per-perturbation residual structure that is not resolvable at this SNR. **The gap is
real (46% of C), but its per-perturbation, recoverable part is obscured by sampling noise at the
pseudobulk level.**

**B2 verdict → RED confirmed, and the data-need sharpened.** The residual comprises (i) a shared
*transient* activation program (condition-level, already captured by any per-condition model, and not
the prediction frontier) plus (ii) a perturbation-specific part that is noise-limited at pseudobulk.
Recovering (ii) requires **single-cell resolution and more cells per perturbation** (to beat the noise
floor and to test within-condition cell-state dependence, the CellCap seam) — that is, a raw-cell
download rather than a more sophisticated pseudobulk model. Output: `results/phaseB_recovery.csv`.

---

## B3 — routed build: route = **RED** (measure-and-stop)  [IN-PROJECT]

**Route decision.** Q3 fires (transition-peaked, 4/4 donors, effect- and confound-controlled), and this
is confirmed by B2 (the recoverable-at-pseudobulk part is a shared transient program, while per-pert
structure is noise). Per the pre-registered routing, **Q3-fires → RED**: a trajectory or
non-autonomous vector-field model is not to be built. Three timepoints (Rest / Stim8hr / Stim48hr)
cannot identify a driven vector field; a long GPU run there would produce a *confident artifact*, which
is worse than nothing. **No build was started, and the GPU was left idle** (per the brief, the box is
left idle rather than manufacturing work when the route is RED). CP2 and the budget were untouched, and
nothing was merged.

**Robustness of the decisive fingerprint  [IN-PROJECT].** The transition-peak is clear in the canonical
CIPHER raw-count residual (`_cipher`: Rest 0.813 < Stim8hr 0.943 > Stim48hr 0.931, peak +0.071). The
`_exact` and log1p (`residual.csv`) variants are **degenerate** — the residual saturates at
approximately 0.995 (CIPHER explains approximately nothing; the log1p case is the normalisation the
sizer docstring flags as broken) — so they carry no fingerprint. Caveat: the fingerprint is a property
of the *meaningful* (raw-count, CIPHER-Methods) residual, and the reconstruction matched it exactly
(corr 1.000).

**What it is → what data it needs.**
- *What it is:* a **transient T-cell activation cytokine program** (IFNG, IL2, CSF2, IL3/IL13,
  CCL/CXCL chemokines, IL2RA, and others) induced across the Rest→Stim transition and peaking at 8hr —
  a far-from-equilibrium response outside the reach of an equilibrium fluctuation-response model. It is
  largely *shared* across perturbations; the *perturbation-specific* residual is noise-limited at
  pseudobulk (cross-donor 0.03).
- *What it needs (for perturbation prediction, the project goal):* **single-cell resolution and more
  cells per perturbation** — in order to (a) lift the per-pert residual above the pseudobulk noise floor
  and (b) test whether it depends on within-condition cell-state (the CellCap seam). This is a raw-cell
  download, **flag-gated** (≈130 GB/stratum), and is **not** to be run unattended.
- *What it does not need:* more timepoints for *prediction* — the transient program is already
  captured at the condition level by per-condition models; denser timepoints would serve *mechanism*
  (identifying the transition vector field), which remains hard and is the RED-forbidden build.

**Pre-registered success bar (stated, not exercised).** Had a build run, it would have "succeeded" only
by recovering ≥25% of the do-operator-adjusted bucket-C remainder (≈46% of C), out-of-sample, while
beating both Ridge and the do-operator. **Because no build ran, no claim is made.** This is a *reported
route* rather than a null-model result.

**Follow-on queue — assessment (no busywork).**
- *Item 1 (routed build):* skipped — RED.
- *Item 2 (do-operator-adjusted per-condition ablation):* the committed do-operator predictions are
  aggregated by hold-out split (gene / condition) and are not resolvable by the 3 biological conditions,
  so a clean per-condition do-operator ablation is **not computable from committed data** (it requires
  per-condition eval predictions that were not committed).
- *Item 3 (second pseudobulk tool):* this would not beat the pseudobulk noise floor (B2), yielding no
  information gain; not run.
- *Item 4 (GREEN sweep):* not applicable (GREEN did not run).

Per the brief, the remaining queue items are exhausted or uninformative on committed data → **stop, GPU
idle.** The one high-value next step (cell download → CellCap test) is flag-gated and is recommended to
the lead below.

---

## State for the lead and recommended next step

**Bottom line.** The bucket-C residual is a **transient T-cell activation cytokine program** — real,
structured, and reproducible (B1) — and it peaks at the Rest→Stim8hr activation transition. However, its
*reproducible* part is a **shared, perturbation-independent** program (captured by any per-condition
model); the **perturbation-specific** residual — the actual prediction frontier — is **noise-limited at
the pseudobulk level** (cross-donor 0.03). The existing state-aware tool (the do-operator) already
leaves approximately 46% of gene bucket C. **Route = RED: no static or trajectory model built on the
current pseudobulk data would honestly clear the bar, and the GPU was left idle rather than filling the
night with a run bound to fail.**

**Recommended next step (FLAG-GATED — requires your authorisation).** Download 1–2 Stim8hr strata of raw
cells (~130 GB each, same-region public S3 `--no-sign-request`, delete-after — the established C-NL
pattern) to run the decisive test that the pseudobulk cannot: (1) whether per-perturbation residual
structure appears above noise at single-cell SNR, and (2) whether it depends on within-condition
cell-state (the **CellCap** seam). If it does, a GREEN CellCap build becomes justified (with no
additional timepoints). If it does not, the residual is genuinely a shared transient program, and the
honest headline is a "predictability ceiling on this axis; the per-pert frontier needs a different
experiment (denser transition sampling)." This was **not** run unattended (>5 GB download flag and RED
routing).

**What was and was not done.** Done: B1 localisation and B2 recovery, committed (`3ef19ca`, `6ea0e65`).
Not done: no model was built, no cells were downloaded, CP2 / the budget / Dev 4's branches were not
touched, and nothing was merged.

**Box status at hand-off.** The GPU is **idle** (0% util); no Phase-B job is running or queued, and the
box is free. Committed artifacts are on branch `phaseB` (unpushed, unmerged). Scratch is on the box at
`~/cd4-phaseB/`.

---

## Step 0 — SNR pre-check for the single-cell CellCap seam → **NOT GREEN (confirmed floor)**  [IN-PROJECT]

The lead greenlit the ~130 GB single-cell run *conditional on* an SNR pre-check: is the ~0.033
per-perturbation reproducibility a **pseudobulk-aggregation artifact that single-cell resolution fixes**,
or a **genuine effect-size/noise floor** that it cannot? This was computed from committed data only,
with no download.

**Decisive fact.** Single-cell modelling of a perturbation's *mean* effect does **not** add cells,
because pseudobulk is the sufficient statistic for the mean; the same ~180 cells/pert set the floor
either way. Single-cell resolution helps only via within-state **concentration** (an effect localised to
a subpopulation) or via higher moments.

| metric | value | reading |
|---|---|---|
| cross-donor reproducibility, full 3000 genes | 0.029 | matches Phase B's ~0.033 |
| cross-donor reproducibility, **relevant top-50 effect genes** | **0.033** | focusing on the genes that matter does **not** lift it |
| perturbations with detectable effect (SNR>3) | **17%** | most per-pert effects are weak at 180 cells |
| within-donor reliability (noise model) | 0.48 | but ≫ cross-donor 0.03 → the non-noise structure is **donor-specific**, not reproducible biology single-cell repairs |
| cells needed to reach usable r=0.30 | **~12×** | not available (experimental cell count is fixed) |
| concentration needed instead | **~8% of cells** | stringent; even a generous f=0.3 projects to only r≈0.10 |

**The single number for the lead: per-perturbation cross-donor reproducibility on the relevant genes =
0.033 (unchanged from the full-vector floor); projected single-cell best-case ≈ 0.10, short of the 0.30
usable floor.** → **NOT GREEN.** The per-perturbation frontier is a genuine effect-size/noise (and
donor-variability) floor, **not** a pseudobulk artifact that single-cell resolution fixes. There is **no
download and no build; the 130 GB is not spent.** The box stays down.

*Honest caveat (the one scenario in which single-cell might help).* The within-donor 0.48 versus
cross-donor 0.03 gap *could* reflect donor differences in cell-state composition (which cell-state
conditioning would disentangle) rather than pure batch. The pre-check cannot separate these without
cells, but the underlying effects are weak (17% detectable, median SNR 1.7), so even in that best case
the expected recovery is low. The recommendation stands: **do not spend the download on an
~8%-concentration bet over weak effects.** If the lead wishes to pursue the composition hypothesis
specifically, that is the only justification — a deliberate low-probability bet rather than the default.

Outputs: `results/phaseB_snr_precheck.csv`, `phaseB_snr_precheck_summary.csv`.

**Result type:** a clean, reported **confirmed floor** ("the per-perturbation frontier is noise-limited
even at single-cell resolution"), which is exactly the pre-registered NOT-GREEN outcome — reached with
no GPU-hours and no egress spent.
