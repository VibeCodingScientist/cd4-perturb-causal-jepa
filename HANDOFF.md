# Team Handoff — CD4+ Perturbation Causal + JEPA

**Status: CP1 complete and submittable** (tag `cp1`). Repo public at
`github.com/VibeCodingScientist/cd4-perturb-causal-jepa`. This doc is the single place to get
oriented: what's built, the result, how it reproduces, and exactly what Developer 2 picks up.

Read alongside: [`UNIFIED_BUILD_PLAN.md`](UNIFIED_BUILD_PLAN.md) (spec), [`README.md`](README.md)
(overview + results), [`hypotheses.md`](hypotheses.md) (pre-registration), [`RUNBOOK.md`](RUNBOOK.md)
(one-command reproduction).

---

## 1. TL;DR

- Predicting CRISPRi knockdown transcriptional effects in primary human CD4+ T cells
  (Marson/Pritchard genome-scale Perturb-seq, GSE278572 / CZI Virtual Cells).
- The experimental core is a **2×2 ablation** (encoder-init × causal-mask). CP1 delivers the
  **random-init row** (causal vs non-causal) + baselines. The JEPA row (CP2) is Developer 2.
- **Headline (pre-registered C2): the do-operator works.** The corrected do-mask beats its
  non-causal twin by **+52%** (condition hold-out) and **+79%** (gene hold-out) on Pearson-δ.
- Everything short of the JEPA cells is built, reviewed (20-agent adversarial pass, 8 bugs
  fixed), and run end-to-end on a real L4 box.

---

## 2. CP1 result (real data, NVIDIA L4)

Frozen split (`split_manifest.json`, SHA `fd2b8c21…` bound to `GWCD4i.pseudobulk_merged.h5ad`):
gene-holdout = 1,729 genes (15%), condition hold-out = Stim48hr, donor probe = donor_4,
3,000 HVG. Evaluable HVG-panel perturbations: **2,269 (condition) / 318 (gene)**.

| split | model | Pearson-δ (top-50) ↑ | PerturBench rank ↓ (>0.4 = collapsed) | DES ↑ | E-dist ↓ |
|---|---|---|---|---|---|
| **condition** | **causal** | **0.344** | 0.457 | 0.587 | 2.46 |
| condition | non-causal | 0.226 | 0.483 | 0.579 | 6.46 |
| condition | ridge | 0.384 | **0.365** | 0.651 | 0.19 |
| condition | fcn | 0.086 | 0.500 | 0.535 | 2.88 |
| **gene** | **causal** | **0.368** | 0.440 | 0.599 | 2.09 |
| gene | non-causal | 0.206 | 0.484 | 0.590 | 5.11 |
| gene | ridge | 0.019 | 0.501 | 0.506 | 0.18 |
| gene | fcn | 0.107 | 0.500 | 0.554 | 2.30 |

Full 8-metric table: `results/benchmark_table_full.csv`.

### How to read it (vs the pre-registered outcomes in `hypotheses.md`)

- **C2 — do-operator isolation (CONFIRMED, the headline).** Causal beats non-causal on *both*
  hold-outs, consistent across Pearson-δ, E-distance, and AUPRC. This is the "do-operator provides
  real inductive bias" outcome we pre-registered as the headline. The corrected mask (perturbed
  gene's query row masked, key column kept so the intervention propagates) is doing the work.
- **Zero-shot to unseen genes:** causal generalizes (0.368) where the linear baseline collapses
  (Ridge 0.019). The gene-token encoder (ESM-2 + proxy tokens) carries real transferable signal.
- **C1 — vs strong baselines (mixed, reported honestly).** On the *pure condition shift*, Ridge
  (0.384) edges out the causal transformer (0.344); on the gene hold-out causal dominates. So the
  causal architecture's advantage is largest for unseen-gene generalization, not the activation-
  state shift — the reverse of one pre-registered guess, and a clean result either way.
- **Mode-collapse honesty:** the detector (§7i) flags every model except Ridge/condition
  (rank < 0.4). The transformers sit at 0.44–0.48 — borderline, and causal is always sharper than
  non-causal. FCN collapses outright (predicts ~mean), exactly the failure mode the detector exists
  to surface. Not hidden.
- **Gates:** TabPFN is license-gated on a headless box (Prior-Labs models need a browser license
  click or a `TABPFN_TOKEN`); reported N/A, addable in one step. C3 (JEPA) is CP2.

---

## 3. What's built + ownership

One importable `core` package; shared artifacts live in `DATA_ROOT` (outside git). The frozen
interface is `core/contract.py` — both developers code against it and never collide.

| Area | Files | Owner | State |
|---|---|---|---|
| Contract (paths, schemas, namespaces, eval sig) | `core/contract.py` | Dev 1 | frozen |
| Data ETL (backed h5ad, CZI pseudobulk adapter, QC, HVG) | `core/data.py` | Dev 1 | done |
| Pseudobulk (streaming accum, matched-control δ, routing) | `core/pseudobulk.py` | Dev 1 | done |
| Features (ESM-2, DEG-frequency, context prior) | `core/features.py` | Dev 1 | done |
| Split (2-phase freeze, SHA verify) | `core/split.py` | Dev 1 | done |
| Eval (frozen metrics + mode-collapse detector) | `core/eval.py` | Dev 1 | frozen |
| Do-attention (corrected mask) | `core/models/do_attention.py` | Dev 1 | done |
| Gene-token encoder (shared w/ JEPA) | `core/models/gene_tokens.py` | Dev 1 | done |
| Causal transformer + non-causal twin | `core/models/causal_cistransformer.py` | Dev 1 | done |
| Baselines (Ridge/TabPFN/FCN) | `core/models/baselines.py` | Dev 1 | done |
| GPU queue (serial single-GPU scheduler) | `gpu_queue.py` | Dev 1 | done |
| Pipeline + env + box scripts | `Snakefile`, `environment.yml`, `scripts/` | Dev 1 | done |
| **JEPA pretraining (EMA teacher, single-cell)** | `core/models/jepa.py` | **Dev 2** | TODO (CP2) |
| **JEPA→causal integration + 2×2 harness** | (new) | **Dev 2** | TODO (CP2) |
| **VOI + subsampling** | `core/voi.py` | **Dev 2** | TODO (CP2) |
| **Figures** | `figures/` | **Dev 2** | TODO (CP2) |

**Tests (6 files, all green locally + on the box):** `test_core_synthetic`,
`test_baselines_synthetic`, `test_do_mask` (proves the mask propagates + clamps; guards against
re-adding `M[:,perturbed]=-inf`), `test_causal_synthetic`, `test_gpu_queue`, `test_fixes`
(regression coverage for the review + real-data fixes).

---

## 4. Reproducibility

- **Data (public S3, no creds):** `s3://genome-scale-tcell-perturb-seq/marson2025_data/`.
  CP1 uses `GWCD4i.pseudobulk_merged.h5ad` (44.6 GB). The 12 per-donor cell h5ads
  (`D*_*.assigned_guide.h5ad`, ~1.7 TB total) are the JEPA input.
- **One-command:** see [`RUNBOOK.md`](RUNBOOK.md) — clone → `environment.yml` → download →
  `build_from_czi_pseudobulk` → `scripts/build_priors.py` (ESM-2) → `scripts/run_cp1.py`.
- **Gene priors:** protein sequences come from the bulk Ensembl peptide FASTA (99.8% coverage).
  Do **not** use `gget.seq` for the whole panel — 3,000 per-gene queries stall.

### The box (CP1 run environment)
- `ubuntu@54.163.21.62` — NVIDIA L4 (24 GB), 16 vCPU / 60 GB RAM, ~1 TB disk.
- Repo at `~/cd4-perturb-causal-jepa`; venv `.venv` (torch 2.12 +cu130); `CD4_DATA_ROOT=/home/ubuntu/cd4-perturb-data`.
- CP1 wall-clock: data build ~7 min, ESM-2 ~14 min, causal+non-causal ~50 min. ~$5–10 of GPU.

---

## 5. Coordination state

- **Branches/tags:** work on `main`. `core-frozen` = the frozen data+eval foundation (Dev 2's
  historical start point). `cp1` = this result. Dev 2 can branch off `main` (has everything).
- **Single GPU:** every GPU job goes through `python gpu_queue.py submit <job>` — atomic lock,
  §6 priority order, epoch-1 measure-then-extrapolate gate. Two worktrees never train at once.
- **`jepa` / `jepa_finetune` job hooks** already exist in `gpu_queue.py`; they dispatch to
  `core.models.jepa.run_jepa` / `run_jepa_finetune` once Dev 2 writes them.

---

## 6. Developer 2 — CP2 starts here

Goal: complete the 2×2 (add `jepa_only`, `jepa_causal`) + VOI/subsampling + the three demo figures.

1. **`core/models/jepa.py`** — implement the exact recipe in `UNIFIED_BUILD_PLAN.md` §7e:
   student on masked input + **EMA teacher (stop-grad) on unmasked** + predictor + cosine loss,
   at **single-cell** resolution (mask expression *values*, ≤600 HVG/cell). Reuse
   `core.models.gene_tokens.GeneTokenEncoder` (same class → weights transfer). Expose
   `run_jepa(**kw)` for the queue. Collapse guard: log teacher-embedding std.
2. **Cells:** subsample ~1–2M cells (stratified donor×condition) from the per-donor h5ads into
   `DATA_ROOT/cells/`. `core.data.stratified_cell_indices` is the sampler. 906 GB free fits this.
3. **Integration:** init the causal encoder from the JEPA checkpoint → fine-tune → writes
   `runs/jepa_only_*.parquet`, `runs/jepa_causal_*.parquet`. Score via `core.eval` (unchanged).
4. **`core/voi.py`** — mean pairwise L2 disagreement across the ensemble; subsampling curve at
   5/10/20/50/100% (§7h). **Figures** (§12): benchmark table (collapse in red), 2×2 bar chart on
   condition-hold-out Pearson-δ (reads out C2 + C3), sample-efficiency curve.
5. Submit JEPA to the shared `gpu_queue.py`; never train while another GPU job runs.

The eval harness, split, features, and encoder are frozen — build against them, don't fork.

---

## 7. Open items / known limitations

- **TabPFN** N/A (license-gated). Add with a Prior-Labs `TABPFN_TOKEN`, then
  `b.run_tabpfn()` appends its rows. It's the one CP1 model missing.
- **FCN collapses** (predicts ~mean). It's a valid flagged baseline but under-tuned; a stronger
  VCC-2nd-place-style FCN would be a better comparator.
- **Context prior** is currently an ESM-2 PCA stand-in. The plan's node2vec-over-STRING network
  prior is wired (`features.build_context_prior_node2vec`) but not yet run.
- **Transformer discrimination** is borderline (rank 0.44–0.48). Worth probing whether more
  training / a sharper prediction setup (per-donor control conditioning) lifts it below 0.4.
- **Ops:** the L4 box is billed hourly — stop it when idle (CP1 done) unless rolling into CP2.

## 8. Supplementary — mechanism recovery (negative result)

Self-contained CPU-only study in [`mechanism/`](mechanism/) — does not touch any CP1/CP2 file. Tests
whether explicit per-context causal-matrix (`Â_C`) estimation beats correlation baselines for
cross-context transportability. **Result: FAIL under `P≪G`**, in both a linear/single-perturbation and a
nonlinear/double-perturbation regime; reported honestly with pre-registered bars. The core finding is
*why* correlation is hard to beat (Lyapunov sufficiency of the control covariance), plus a standalone
result that the linear transportability condition degrades AUROC 1.00 → 0.88 under nonlinearity.
Reproduce: `python run_c4.py && python eval.py && python sensitivity.py` (spike 1);
`python run_spike2.py && python spike2_diag.py` (spike 2). Full readout: [`mechanism/README.md`](mechanism/README.md).

A third probe — the **C-NL gate** (`python run_cnl_gate.py`) — is the **positive** of the line: on ground
truth the baseline **third moment predicts the second-order response covariance provably misses** (ΔR²
≈ +0.6–0.75, CI excluding 0; survives to 1,000 cells with NB emission on). The term is small (~3–4%) but
structured; the open go/no-go is sizing the analogous residual on real CD4 (within-donor, stratified by
effect size — the large-effect bin is decision-relevant). Readout: [`mechanism/FINDINGS_CNL.md`](mechanism/FINDINGS_CNL.md).
