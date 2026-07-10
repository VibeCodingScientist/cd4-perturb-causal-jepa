# C-FUSE Phase 1 — External validation of the do-operator (bounded readout)

*Developer 2, autonomous. Does the do-operator's genome-scale causal signal replicate causal edges
measured independently in other CD4 datasets? Phase 1 is CPU-only, no build, nothing merged; the frozen
submission (`submission-fallback-v1` / `6476670`) and CP2 are untouched. **Outcome: G-F.1 passes on edge
count, but the non-circular external-validation test (G-F.2) is not achievable at the frozen split's
held-out coverage — it reduces to 2 regulators — and the frozen do-operator predictions it needs were
overwritten on the box. A bounded, honest result the lead can act on cold.***

## Source-of-truth corrections (repo/paper win over the brief)
- The brief's **"emdann 2024"** is actually **Weinstock et al. 2024, Cell Genomics** (LLCB). Corrected.
- The brief's **"211 edges over 84 TFs" is [VERIFIED] correct** — 350/211/151 at |β|≥0.020/0.025/0.030,
  matching the paper exactly (`data/external/PROVENANCE.md`).
- **Both external papers are Marson/Pritchard-consortium** — same lab as our GSE278572. So this is
  cross-**platform/modality** validation (our CRISPRi Perturb-seq vs their CRISPR-KO + LLCB / bulk
  RNA-seq), **not** fully independent-lab replication. Any "external validation" claim must say so.

## Step 0 — external edge tables obtained ✅ (both)
- **Weinstock Table S1** (`mmc2.xlsx`): 6,972 β-signed parent→child pairs among 84 TFs → **211** direct
  causal edges at |β|≥0.025. [VERIFIED]
- **Freimer Suppl. Data 1** (RNA_Seq_results.csv, 269,080 rows): **50,654** regulator→target DE edges at
  FDR<0.05 (logFC sign = direction), 24 regulators. [VERIFIED] (includes indirect DE, unlike Weinstock's
  direct edges — kept separate.)
- URLs, dates, row counts, read-tiers in `data/external/PROVENANCE.md`. Neither failed → both used.

## G-F.1 — measurable overlap: **PASS on the ≥50 bar, but see the binding limit**
Measurable shared edge = external edge (regulator R → target T) where R is in our **assayed
perturbations** AND T is in our **HVG readout**. Held-out (non-circular) additionally requires R in the
frozen `gene_holdout`.

| set | edges | measurable | measurable regulators | **held-out measurable** | **held-out regulators** |
|---|---|---|---|---|---|
| Weinstock (direct) | 211 | 104 | 22 | 11 | 2 |
| Freimer (DE) | 50,654 | 15,438 | 24 | 1,534 | 2 |
| **combined** | 50,865 | **15,542** | 28 | 1,545 | **2** |

**G-F.1 verdict: PASS** — 15,542 combined measurable edges ≫ 50. **But the binding constraint for the
valuable (non-circular) test is not the edge count — it is the held-out *regulator* count, which is 2**
(**ETS1**, **PTEN**): the frozen 15% gene-holdout caught only 2 of the 28 measurable external regulators.
1,545 held-out edges sound ample, but they are the targets of just 2 regulators, so the effective
independent units for a non-circular test are ~2, not 1,545.

## G-F.2 — do-operator recovery of held-out external edges: **NOT RUN (double blocker)**
The pre-registered G-F.2 (do-operator sign/direction recovery > non-causal twin > degree-preserving null,
p<0.01, **held-out regulators only**) is not achievable in Phase 1 for two independent reasons:

1. **Held-out coverage = 2 regulators.** A degree-preserving null and a p<0.01 bar over the target sets
   of 2 regulators (ETS1, PTEN) cannot yield a statistically meaningful, non-circular result — regardless
   of the predictions. This alone bounds the achievable Phase-1 headline.
2. **The frozen do-operator predictions are unavailable on the box.** `runs/causal_gene.parquet` and
   `runs/noncausal_gene.parquet` (the do-operator and its twin) are currently **(18×200) synthetic
   fixtures** (fake sequential IDs), **overwritten 2026-07-10** — almost certainly by test-suite runs
   writing to the real `CD4_DATA_ROOT/runs/` during the freeze verification (`test_causal_synthetic` /
   `test_baselines_synthetic`). Only the **rejected v2 challenger** (`*_v2`, real 318×3000) and the
   synthetic fixtures remain; the frozen predictions would need a **GPU regen** (out of Phase-1 scope),
   and the v2 model is the wrong (rejected) model to substitute. **The committed CP2 numbers
   (`benchmark_table.csv`) are unaffected** — only the ephemeral `DATA_ROOT` scratch predictions were hit.

**G-F.2 verdict: not run.** Not a scientific negative — a coverage + data-availability limit, reported
honestly rather than forced on synthetic/rejected/regenerated data.

## Honest framing
- The **edge overlap exists and is large** (15,542 measurable), so the fusion has a surface in principle.
- The **non-circular external-validation result the brief sought is not establishable** at the frozen
  split's held-out coverage (2 regulators) — that is the real Phase-1 finding, independent of the clobber.
- Even if run, recovering these edges would be **validation, not discovery**, and against **same-lab**
  data. Keep any claim at that.

## What is untouched
Frozen tag `submission-fallback-v1` / `6476670` and CP2 `benchmark_table.csv` are byte-untouched. No GPU
spent. No build (G15 fusion / G-F.3 donor-variance) entered. Nothing merged to `main`. Box idle.

## Recommended next step for the lead (awake decision)
The cross-dataset edge overlap is ample, but the non-circular test is blocked by (a) only 2 held-out
external regulators under the frozen split and (b) clobbered predictions. Options:
1. **Regenerate** `runs/causal_gene.parquet` + `noncausal_gene.parquet` (1 GPU inference of the frozen
   CP2 models on the 318 held-out perts) and run a **2-regulator (ETS1, PTEN)** G-F.2 — but flag it as
   underpowered (2 regulators) and same-lab.
2. **Re-scope** to a purpose-built **regulator-holdout** that deliberately holds out more of the ~90
   external regulators (changes the split — a new experiment, not the frozen one).
3. **Fold as a bounded finding** — "measurable cross-dataset edge overlap is large (15.5k edges), but the
   non-circular validation is coverage-limited to 2 held-out regulators under the current split" — and
   move on. No further action needed.
Also worth a quick fix regardless: point the test suite at a temp `DATA_ROOT` so tests can't clobber real
`runs/` predictions again.

Deliverables: `results/fusion_gates.csv` (= overlap table), `results/fusion_measurable_edges.csv`,
`data/external/weinstock_edges.csv`, `data/external/PROVENANCE.md`, `figures/fusion_gates.png`.
