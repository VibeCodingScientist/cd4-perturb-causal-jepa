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

- **GEO:** GSE278572 · CZI Virtual Cells mirror
- **Split SHA256:** _(recorded here once `split_manifest.json` is frozen against the
  downloaded `.h5ad` — every module verifies this hash at startup)_

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

## License

[MIT](LICENSE).
