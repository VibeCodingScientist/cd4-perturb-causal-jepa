# C-DON — Donor-Structured Recovery gate → **NO-GO (fifth clean negative; floor confirmed)**

*Developer 2. A CPU-first gate that could have **revised** a submission headline: the current
conclusion is "the per-perturbation frontier is a noise floor," but one committed number — per-
perturbation reproducibility **0.48 within-donor vs 0.03 cross-donor**, a 16× gap noise cannot make —
contradicted it. This gate tested, with the paper's own two-guide design, whether that gap is real
recoverable donor-specific biology (→ partial reversal) or an artifact. CP2/budget frozen; no GPU spent.*

**Provenance:** [IN-PROJECT] measured here on committed data · [VERIFIED] against the primary (Marson
GSE278572 / bioRxiv 2025.12.23.696273): 2 gRNAs/gene, pooled genome-scale library · [INFERENCE] under test.

## Step 0 — pre-checks (both cleared; the gate is runnable and valid)

- **0a Guide-data regime:** the committed pseudobulk is **per-guide** (`guide_id` = `GENE-1`/`GENE-2`;
  25,954 guides / 12,731 genes ≈ **2.04/gene**). Guide identity is retained → G-D.1 is a CPU script,
  no raw-h5ad recompute. `[IN-PROJECT/VERIFIED]`
- **0b Co-location confound (load-bearing):** the 2 guides per gene are **batch-orthogonal** —
  **12,510/12,731** genes have guides in *both* 10x runs; each run holds ~all genes (pooled screen).
  Within-donor same-gene concordance therefore **cannot be faked by shared batch**. `10xrun_id` kept as
  an explicit covariate (Test B). **Gate valid.** `[IN-PROJECT]`

## G-D.1 — biology vs batch (population-level, per-donor): **NO-GO**

Same-gene independent-guide within-donor concordance vs different-gene, on the perturbation-*specific*
effect (shared program removed), pooled over conditions:

| donor | same-gene (specific) | diff-gene | Δ | Δ (composition-corrected) |
|---|---|---|---|---|
| CE0006864 | 0.0134 | −0.0030 | 0.0164 | 0.0223 |
| CE0008162 | 0.0127 | +0.0011 | 0.0116 | 0.0145 |
| CE0008678 | 0.0113 | −0.0066 | 0.0179 | 0.0173 |
| CE0010866 | 0.0124 | +0.0002 | 0.0122 | 0.0219 |
| **median** | **~0.013** | **~−0.003** | **~0.014** | **~0.020** |

- **4/4 donors positive; permutation p < 0.001** (specific *and* composition-corrected).
- **Survives composition-correction** (Δ actually rises when the activation axis is removed) → the
  signal is not donor activation-state composition.
- **Test B:** effect magnitude does not track batch — corr(|s|, 10xrun) = **−0.081** (depth/n-cells
  correlate with |s| only in magnitude, the expected noise-shrinks-with-cells direction).

**Verdict — NO-GO.** There *is* real, statistically significant, composition-robust, batch-excluded
**target-specific biology** (independent guides of the same gene agree above chance). **But its
magnitude is at the noise floor:** Δ ≈ 0.014–0.020 is **~8× below the pre-registered Δ ≥ 0.15 GO bar**,
and the absolute same-gene concordance (~0.013) means two independent guides share only ~1% of their
specific effect. Not "batch" (batch excluded), not underpowered (9–21k same-gene pairs/donor), not
composition (survives) — it is **real biology at noise-floor magnitude.**

## G-D.2 — donor-conditioning recovery: **NO-GO**

Predict an independent held-out guide g2's within-donor specific effect from: donor-CONDITIONED
(same-donor g1), donor-AVERAGED (cross-donor consensus), predict-the-donor-MEAN, and donor-PERMUTED
(wrong-donor g1). The donor-conditioned term *is* the same-gene within-donor concordance.

(n = 22,233 gene×donor pairs)

| predictor | held-out corr | vs bar |
|---|---|---|
| donor-CONDITIONED (same-donor g1→g2) | **0.0157** | ≪ 0.20 |
| **donor-AVERAGED (cross-donor consensus)** | **0.0339** | — |
| predict-the-donor-MEAN | 0.0090 | — |
| donor-PERMUTED (wrong-donor g1) | 0.0237 | — |

**Verdict — NO-GO, and decisively so.** Donor-conditioned recovery (0.016) is **≪ the 0.20 bar** — but
the sharper point is that **donor-AVERAGING (0.034) *beats* donor-CONDITIONING (0.016)**, and even the
donor-PERMUTED (wrong-donor) predictor (0.024) beats it. **Donor-conditioning provides *negative* gain:
averaging over donors denoises the estimate and predicts a held-out guide better than any same-donor
signal.** This is the exact opposite of the reversal hypothesis — there is **no** recoverable
per-perturbation donor structure, and the cross-donor consensus the four negatives used is in fact the
*best* predictor. The build is not licensed.

## The key finding (why this resolves the tension)

**The "0.48 within-donor" was a noise-*model*-predicted reliability, not an empirical measurement.**
Measured empirically with **independent guides**, within-donor per-perturbation concordance is
**~0.016 ≈ the noise floor**, and cross-donor *averaging* actually predicts a held-out guide **better**
(0.034 vs 0.016). **The 16× gap does not just evaporate — it inverts:** averaging *helps*. The
per-perturbation frontier is a **real noise floor where donor-averaging is beneficial**, not a
donor-averaging artifact — confirmed by the paper's own two-guide design.

## Routing & honest ceilings

**G-D.1 NO-GO → fifth clean negative → STOP.** The submission's noise-floor headline **holds, and is
strengthened**: the reversal hypothesis is refuted, and the misleading 0.48 is explained. Joins F1
(Â_C), F2 (fluctuation/3rd-moment), F3 (CellCap SNR), F4 (trajectory-geometry) as dead directions.

- **No generalizable-donor claim, no method-novelty claim** (donor-as-covariate is standard). The
  defensible core is the in-project logic on the flagship dataset.
- **The build (G13) is NOT licensed.** No GPU spent; nothing merged; own worktree `donor-gate`.

Deliverables: `results/donor_structure_gate.csv`, `results/donor_structure_gd2_*.csv`,
`figures/donor_structure_gate.png`, `hypotheses.md` (C-DON pre-registration + result).
