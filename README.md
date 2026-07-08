# CD4+ T Cell Perturbation Prediction — Causal + JEPA

**Built with Claude: Life Sciences (2026).**

Predicting the transcriptional effect of a CRISPRi gene knockdown in primary human
CD4+ T cells — including in an **activation state the model has never seen**.

> "Standard models treat a gene knockdown as an *observation*. We treat it as an
> *intervention* — and that distinction is what lets the model predict a
> knockdown's effect in an activation state it has never seen."

This repository is one pipeline whose experimental core is a **2×2 ablation**
(`JEPA-init × causal-mask`). The causal claim, the do-operator isolation, and the
JEPA claim are three cells of that single matrix. Ridge / TabPFN / PseudoBulk-FCN /
Arc State are external reference points; a Value-of-Information (VOI) layer turns
model disagreement into a sample-efficient experimental-design recommendation.

The full technical specification is [`UNIFIED_BUILD_PLAN.md`](UNIFIED_BUILD_PLAN.md).
The pre-registered hypotheses are in [`hypotheses.md`](hypotheses.md) and were
committed **before** any model saw data.

---

## Dataset

Marson/Pritchard genome-scale CRISPRi Perturb-seq in primary human CD4+ T cells
(~22M cells, every expressed gene silenced one at a time, 4 donors, 3 activation
states: Rest / Stim8hr / Stim48hr).

- **GEO:** GSE278572 · CZI Virtual Cells mirror · bioRxiv `10.64898/2025.12.23.696273`
- **CP1 data source:** `GWCD4i.pseudobulk_merged.h5ad` (44.6 GB pre-computed pseudobulk;
  278,684 guide×donor×condition profiles × 18,129 genes). CP1 runs on pseudobulk deltas;
  the ~1.7 TB of single cells are the JEPA lane's input.
- **Split SHA256 (frozen):** `fd2b8c21d357f8699ec34e2d5ebc1639612c27a0147a9ca94d4983822d93247e`
  — binds the split to that exact file; every module verifies it at startup.

## The claims (pre-registered — see [`hypotheses.md`](hypotheses.md))

| ID | Claim |
|----|-------|
| **C1** | The `CausalCisTransFormer` (corrected do-mask) matches or beats strong baselines **including TabPFN** on the **condition hold-out** (zero-shot Stim48hr). |
| **C2** | The causal mask beats its non-causal twin on the condition hold-out (do-operator isolation). Reported regardless of leaderboard position. |
| **C3** | JEPA-init helps the condition hold-out (full 2×2, JEPA × causal). |
| **S1** | Model-disagreement VOI ranks which perturbations are most worth measuring; VOI-guided selection reaches ~90% of full-screen accuracy from a fraction of perturbations. |

**The 2×2 (experimental core):**

| Encoder init | Causal mask | Label |
|---|---|---|
| Random | off | Direct-regression baseline (`noncausal`) |
| Random | on  | Causal-only — C2 treatment (`causal`) |
| JEPA   | off | JEPA-only (`jepa_only`) |
| JEPA   | on  | **JEPA + causal — main model** (`jepa_causal`) |

## CP1 results (real data, L4)

Trained on the frozen split (SHA above): 2,269 / 318 evaluable HVG-panel perturbations on the
condition / gene hold-outs. Headline metrics (`results/benchmark_table.csv`; full 8-metric
appendix in `results/benchmark_table_full.csv`):

| split | model | Pearson-δ (top-50) ↑ | PerturBench rank ↓ | DES ↑ |
|---|---|---|---|---|
| **condition** | **causal** | **0.344** | 0.457 | 0.587 |
| condition | non-causal | 0.226 | 0.483 | 0.579 |
| condition | ridge | 0.384 | **0.365** | 0.651 |
| condition | fcn | 0.086 | 0.500 | 0.535 |
| **gene** | **causal** | **0.368** | 0.440 | 0.599 |
| gene | non-causal | 0.206 | 0.484 | 0.590 |
| gene | ridge | 0.019 | 0.501 | 0.506 |
| gene | fcn | 0.107 | 0.500 | 0.554 |

**C2 — the do-operator works (headline, pre-registered).** Same architecture, mask on vs off:
the causal mask beats its non-causal twin by **+52%** on the condition hold-out (0.344 vs 0.226)
and **+79%** on the gene hold-out (0.368 vs 0.206). The advantage is consistent across
Pearson-δ, E-distance (2.46 vs 6.46 condition; lower better), and AUPRC. The corrected
do-mask — masking only the perturbed gene's query row so the intervention propagates
downstream — is doing real work.

**Zero-shot to unseen genes:** the causal model generalizes to genes it never saw silenced
(**0.368**) where the linear baseline fully collapses (Ridge **0.019**).

**Honest caveats.** On the pure condition shift, a simple gene→δ linear map (Ridge, 0.384) is
still competitive with / slightly ahead of the causal transformer (0.344), and only Ridge on the
condition hold-out clears the mode-collapse bar (rank < 0.4) — the transformers sit in the
borderline 0.44–0.48 band (causal always sharper than non-causal). TabPFN is license-gated on a
headless box (see RUNBOOK); the JEPA cells of the 2×2 (C3) are CP2.

## The two corrections this build refuses to revert (`UNIFIED_BUILD_PLAN.md` §1)

1. **Causal do-mask propagates.** An intervention removes only edges *into* the
   perturbed gene (mask its query row); other genes **must still attend to it** so
   the intervention propagates downstream. We do **not** add `M[:, perturbed] = -inf`.
   (DoFormer, bioRxiv 2026.05.02.722054.)
2. **JEPA uses an EMA teacher at single-cell resolution.** Student on masked input
   + stop-gradient EMA teacher on unmasked input + predictor head + cosine loss,
   masking expression *values* within a cell — not a pseudobulk MLP.
   (Cell-JEPA, arXiv 2602.02093.)

## Repository layout

```
cd4-perturb-causal-jepa/
  core/
    contract.py             # paths + schemas both worktrees code against (frozen FIRST)
    data.py                 # backed h5ad reads, QC, HVG, subsampling
    pseudobulk.py           # (pert,cond,donor) mean profiles + deltas
    features.py             # ESM-2 + network/GO priors, DEG-frequency features
    split.py                # writes/loads split_manifest.json, verifies SHA
    eval.py                 # FROZEN metrics + mode-collapse detector
    models/
      do_attention.py       # corrected DoAttention (§7d)
      gene_tokens.py         # CisTransCell-style gene-token encoder (§7d)
      causal_cistransformer.py
      jepa.py               # Cell-JEPA-style pretraining (§7e) — Developer 2
      baselines.py          # Ridge, TabPFN, PseudoBulk FCN
    voi.py                  # ensemble-disagreement VOI — Developer 2
  results/                  # benchmark_table.csv (committed)
  figures/                  # demo figures (committed)
  hypotheses.md             # pre-registration (committed before Day 1)
  split_manifest.json       # immutable split (committed)
  gpu_queue.py              # single-GPU serial job scheduler (§6)
  Snakefile
  environment.yml
```

**Shared artifacts live outside git** in `DATA_ROOT` (default `~/cd4-perturb-data/`,
override with `CD4_DATA_ROOT`): `embeddings/`, `pseudobulk/`, `features/`, `cells/`,
`runs/`, `checkpoints/`, and the `GPU_LOCK`. Only code, `split_manifest.json`,
`hypotheses.md`, `results/benchmark_table.csv`, and `figures/` are committed. See
`core/contract.py` for every canonical path.

## Reproducing

```bash
# 1. environment
conda env create -f environment.yml && conda activate cd4-perturb

# 2. run the whole pipeline (Lane C on CPU; GPU jobs go through the queue)
snakemake --cores all

# GPU training/inference is never launched directly — always via the serial queue:
python gpu_queue.py submit esm2      # G1
python gpu_queue.py submit causal    # G2
python gpu_queue.py submit noncausal # G3
# ... in the §6 priority order; each job runs the epoch-1 measure-then-extrapolate gate first.
```

## Single-GPU concurrency (`UNIFIED_BUILD_PLAN.md` §6)

Three lanes, one L4 (24 GB). **Lane C** (CPU) runs data/QC/pseudobulk/features/eval
continuously and scores each model the moment its `runs/*.parquet` lands. **Lane G**
is a *serial* GPU queue (`gpu_queue.py`) — one job at a time, in priority order, each
gated by a 1-epoch measure-then-extrapolate check. **Lane D** is code-writing
concurrency across up to two `git worktree` checkouts that share the one queue.

## Acceptance checkpoints

- **CP1** — `results/benchmark_table.csv` has `ridge`, `tabpfn`, `fcn`, `causal`,
  `noncausal` on gene + condition hold-out; split SHA verified; eval harness passing.
- **CP2** — 2×2 complete (`jepa_only`, `jepa_causal`); VOI + subsampling curve; three
  demo figures; `snakemake --cores all` end-to-end.
- **CP3** — reproducibility package tagged; full 8-metric appendix; one-command rerun.

## Supplementary — mechanism recovery

A reproducible synthetic study ([`mechanism/`](mechanism/), CPU-only) testing whether explicit
per-context causal-matrix (`Â_C`) estimation beats correlation baselines for cross-context
transportability. **Result: it does not, under `P≪G`** — in either the linear/single-perturbation or
the nonlinear/double-perturbation regime (pre-registered bar not met → FAIL, honestly reported). It
documents *why* correlation baselines are so hard to beat on the field's own simulator (the stationary
covariance solves the Lyapunov equation, so `Σ` is a near-sufficient statistic for `A`), and — a
standalone positive — how the linear transportability condition itself degrades (AUROC 1.00 → 0.88) as
the system becomes nonlinear. The transportability signal is real (oracle with true `A` = 1.0) but
estimation-gated; the one un-tested lever is a materially better estimator.

A **third probe (the C-NL gate) is the positive** of the line: covariance/Lyapunov sufficiency is a
*second-moment* property, so the one signal it provably cannot carry lives in the third moment. On
ground truth the baseline **third moment predicts the second-order perturbation response covariance
misses** — ΔR² ≈ +0.6–0.75 (CI excluding 0), surviving to 1,000 control cells with NB emission on. The
term is small (~3–4% of the response) but strongly structured; sizing it on real CD4 data is the open
go/no-go. Provenance-guarded (the third-moment link is an inference from response theory, not a CIPHER
claim). See [`mechanism/README.md`](mechanism/README.md) and [`mechanism/FINDINGS_CNL.md`](mechanism/FINDINGS_CNL.md).

## License

[MIT](LICENSE).
