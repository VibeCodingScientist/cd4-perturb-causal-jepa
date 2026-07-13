# C-DON — Donor-Structured Recovery gate → **NO-GO (fifth clean negative; floor confirmed)**

*Developer 2. A CPU-first gate with the potential to revise a submission headline. The current
conclusion is that the per-perturbation frontier is a noise floor, yet one committed number — per-
perturbation reproducibility of 0.48 within-donor versus 0.03 cross-donor, a 16× gap that noise alone
cannot produce — appeared to contradict it. This gate evaluated, using the paper's own two-guide design,
whether that gap reflects genuinely recoverable donor-specific biology (implying partial reversal) or an
artifact. CP2 and its budget remain frozen, and no GPU resources were expended.*

**Provenance:** [IN-PROJECT] measured here on committed data · [VERIFIED] against the primary (Marson
GSE278572 / bioRxiv 2025.12.23.696273): 2 gRNAs/gene, pooled genome-scale library · [INFERENCE] under test.

## Step 0 — pre-checks (both cleared; the gate is runnable and valid)

- **0a Guide-data regime:** the committed pseudobulk is *per-guide* (`guide_id` = `GENE-1`/`GENE-2`;
  25,954 guides / 12,731 genes ≈ 2.04/gene). Because guide identity is retained, G-D.1 can be run as a
  CPU script, with no raw-h5ad recompute required. `[IN-PROJECT/VERIFIED]`
- **0b Co-location confound (load-bearing):** the 2 guides per gene are batch-orthogonal, since
  12,510/12,731 genes have guides in *both* 10x runs and each run holds approximately all genes (pooled
  screen). Within-donor same-gene concordance therefore cannot be produced by shared batch. `10xrun_id`
  is retained as an explicit covariate (Test B). The gate is valid. `[IN-PROJECT]`

## G-D.1 — biology vs batch (population-level, per-donor): **NO-GO**

Same-gene independent-guide within-donor concordance versus different-gene, evaluated on the
perturbation-*specific* effect (with the shared program removed) and pooled over conditions:

| donor | same-gene (specific) | diff-gene | Δ | Δ (composition-corrected) |
|---|---|---|---|---|
| CE0006864 | 0.0134 | −0.0030 | 0.0164 | 0.0223 |
| CE0008162 | 0.0127 | +0.0011 | 0.0116 | 0.0145 |
| CE0008678 | 0.0113 | −0.0066 | 0.0179 | 0.0173 |
| CE0010866 | 0.0124 | +0.0002 | 0.0122 | 0.0219 |
| **median** | **~0.013** | **~−0.003** | **~0.014** | **~0.020** |

- All 4/4 donors are positive, with permutation p < 0.001 (both specific and composition-corrected).
- The signal survives composition-correction: Δ in fact rises when the activation axis is removed, so
  the signal is not attributable to donor activation-state composition.
- **Test B:** effect magnitude does not track batch — corr(|s|, 10xrun) = −0.081. Depth and n-cells
  correlate with |s| only in magnitude, consistent with the expected direction in which noise shrinks as
  cell count increases.

**Verdict — NO-GO.** There is real, statistically significant, composition-robust, batch-excluded
target-specific biology: independent guides of the same gene agree above chance. However, its magnitude
sits at the noise floor. Δ ≈ 0.014–0.020 is approximately 8× below the pre-registered Δ ≥ 0.15 GO bar,
and the absolute same-gene concordance (~0.013) implies that two independent guides share only about 1%
of their specific effect. The effect is not batch (batch is excluded), not a power limitation (9–21k
same-gene pairs/donor), and not composition (it survives correction); it is real biology at noise-floor
magnitude.

## G-D.2 — donor-conditioning recovery: **NO-GO**

An independent held-out guide g2's within-donor specific effect is predicted from four sources: donor-
CONDITIONED (same-donor g1), donor-AVERAGED (cross-donor consensus), predict-the-donor-MEAN, and donor-
PERMUTED (wrong-donor g1). The donor-conditioned term is the same-gene within-donor concordance.

(n = 22,233 gene×donor pairs)

| predictor | held-out corr | vs bar |
|---|---|---|
| donor-CONDITIONED (same-donor g1→g2) | **0.0157** | ≪ 0.20 |
| **donor-AVERAGED (cross-donor consensus)** | **0.0339** | — |
| predict-the-donor-MEAN | 0.0090 | — |
| donor-PERMUTED (wrong-donor g1) | 0.0237 | — |

**Verdict — NO-GO, and decisively so.** Donor-conditioned recovery (0.016) falls well below the 0.20 bar,
but the sharper observation is that donor-AVERAGING (0.034) exceeds donor-CONDITIONING (0.016), and even
the donor-PERMUTED (wrong-donor) predictor (0.024) exceeds it. Donor-conditioning therefore provides
*negative* gain: averaging over donors denoises the estimate and predicts a held-out guide better than
any same-donor signal. This is the exact opposite of the reversal hypothesis. There is no recoverable
per-perturbation donor structure, and the cross-donor consensus used by the four prior negatives is in
fact the best predictor. The build is not licensed.

## The key finding (why this resolves the tension)

The 0.48 within-donor value was a reliability predicted by a noise *model*, not an empirical measurement.
Measured empirically with independent guides, within-donor per-perturbation concordance is approximately
0.016, at the noise floor, while cross-donor *averaging* predicts a held-out guide better (0.034 versus
0.016). The 16× gap does not merely evaporate; it inverts, in that averaging helps. The per-perturbation
frontier is a real noise floor at which donor-averaging is beneficial, not a donor-averaging artifact — a
conclusion confirmed by the paper's own two-guide design.

## Routing & honest ceilings

**G-D.1 NO-GO → fifth clean negative → STOP.** The submission's noise-floor headline holds, and is
strengthened: the reversal hypothesis is refuted, and the misleading 0.48 is explained. This joins F1
(Â_C), F2 (fluctuation/3rd-moment), F3 (CellCap SNR), and F4 (trajectory-geometry) as dead directions.

- There is no generalizable-donor claim and no method-novelty claim (donor-as-covariate is standard).
  The defensible core is the in-project logic on the flagship dataset.
- The build (G13) is not licensed. No GPU resources were expended, nothing was merged, and the work
  remains in its own worktree `donor-gate`.

Deliverables: `results/donor_structure_gate.csv`, `results/donor_structure_gd2_*.csv`,
`figures/donor_structure_gate.png`, `hypotheses.md` (C-DON pre-registration + result).
