# CD4+ T Cell Perturbation Prediction — Unified Build Plan
**One repository, one pre-registration, and three claims, executed on a single NVIDIA L4 (24 GB) with parallel CPU and GPU lanes.**
Dataset: the Marson/Pritchard genome-scale CRISPRi Perturb-seq screen in primary human CD4+ T cells (~22M cells, with every expressed gene silenced one at a time, across 4 donors and 3 activation states: Rest, Stim8hr, and Stim48hr). GEO GSE278572; CZI Virtual Cells mirror.
This plan merges the former "v3 (causal + VOI)" and "v4 (JEPA + causal)" plans. Because they overlapped approximately 80%, they are now consolidated into a single pipeline whose experimental core is a **2×2 ablation (JEPA-init × causal-mask)**. The causal claim, the do-operator isolation, and the JEPA claim correspond to three cells or rows of that single matrix. TabPFN, Ridge, PseudoBulk-FCN, and Arc State serve as external reference points; VOI constitutes the applied layer.
---
## 0. How to use this document
This document is written to drive **one or two Claude Code sessions**. If two are used, they should be run as `git worktree` checkouts of the same repository (see §5). The build is structured so that a submittable result exists early and everything thereafter is additive.
Kickoff prompts are provided in §13. Read §2 (claims), §3 (corrections, which must not be re-introduced), §4–5 (layout and contract), and §6 (concurrency) before writing any code.
---
## 1. Corrections carried in from the plan review (authoritative — do not revert)
Both prior plans contained two errors that would silently break the centrepiece. These are corrected below and must remain corrected.
1. **The causal do-mask over-masks (a do-calculus error).** Following DoFormer (Karbalayghareh et al., bioRxiv 2026.05.02.722054), an intervention removes only the edges *into* the perturbed gene: the perturbed gene ceases to attend to others, but **other genes must still attend to it** so that the intervention propagates downstream. The prior plans added `M[:, perturbed] = -inf`, which severs the outgoing edges and deletes the very signal being predicted. **That line must be deleted.** See §7d for the corrected `DoAttention`.
2. **The JEPA recipe collapses as originally written.** Following Cell-JEPA (arXiv 2602.02093), JEPA requires a **student on masked input, an EMA teacher on unmasked input (with stop-gradient), a predictor head, and a cosine loss**, and it operates at **single-cell** resolution (masking expression *values* within a cell) rather than on a pseudobulk MLP. The prior specification ("no EMA, plain L2, pseudobulk") is a collapse path and discards the 22M cells. See §7e.
Additional corrections:
- **TabPFN version.** Use TabPFN v2 (≤500 features, ≤10k rows) or TabPFN-3 (≤1M rows, ≤200 features). The prior "1024 samples / 50 dims" specification corresponds to v1. Mirror the protocol of the Tabular-Foundation-Models perturbation paper (bioRxiv 2026.06.28.735106), which already applied TabPFN to a genome-wide CRISPR screen in primary human CD4+ T cells and found that tabular models outperform specialised ones on pseudobulk. The novel contribution here is therefore specifically the **condition hold-out combined with causal structure**, not pseudobulk accuracy in general.
- **Arc State.** State is validated for *context transfer*, not for zero-shot unseen perturbations within a cell line (STRAND, arXiv 2602.10156). It will most likely be necessary to train an ST model on the CD4+ train split (transfer, not pure zero-shot). Retain the gate that reduces State to "N/A — resource limit" if it will not run cleanly.
- **Gene-token priors.** Prefer ESM-2 (a function/coding prior) combined with a network/GO prior (a regulatory/context prior, e.g. a GenePT text embedding or a STRING node2vec vector) over DNABERT-2 and Nucleotide Transformer. This combination is lighter on a single L4 and directly addresses the goal of representing "a gene never silenced" by way of function and network neighbourhood. DNABERT-2 and NT are optional and reserved for spare time.
- **Stacking.** Do not K-fold-stack the transformer, as this is too expensive on a single GPU. Stack only the inexpensive base models (Ridge, TabPFN, FCN) using out-of-fold predictions, and treat the transformer as a standalone comparator.
- **Demo claim hygiene.** Remove the "6 months computational exclusivity" claim. State instead that the model was "recently released (Dec 2025); among the first models built on it."
---
## 2. Pre-registration (write `hypotheses.md` and commit BEFORE Day 1)
**Claims**
- **C1 (causal, external).** The CausalCisTransFormer with the corrected do-mask matches or exceeds strong baselines, **including TabPFN**, on the **condition hold-out** (zero-shot Stim48hr).
- **C2 (do-operator isolation).** The causal mask exceeds its non-causal twin on the condition hold-out (a 2×2 row contrast). This is reported regardless of leaderboard position.
- **C3 (JEPA).** JEPA-init improves performance on the condition hold-out; reported across the full 2×2 (JEPA × causal).
- **S1 (support / VOI).** Model-disagreement ranks which perturbations are most worth measuring; VOI-guided selection reaches approximately 90% of full-screen accuracy from a fraction of the perturbations.
**The 2×2 (experimental core).** Encoder-init × causal-mask:
| Encoder init | Causal mask | Label |
|---|---|---|
| Random | off | Direct-regression baseline |
| Random | on | Causal-only (= C2 treatment) |
| JEPA | off | JEPA-only |
| JEPA | on | **JEPA + causal (main model)** |
**Pre-computed outcome interpretations (recorded now, so that Day 5 is analysis rather than improvisation):**
| Outcome | Interpretation |
|---|---|
| Causal beats non-causal on condition hold-out (any init) | The do-operator provides genuine inductive bias under activation-state shift. Headline result. |
| JEPA-init beats random-init on condition hold-out | Representation robustness transfers to cross-context perturbation delta — this **extends Cell-JEPA**, which tested only within a single cell line. To be reported prominently. |
| JEPA helps absolute-state but not gene-hold-out delta | This **corroborates Cell-JEPA's "complementary aspects" finding.** A clean positive-or-null result, and still publishable. |
| TabPFN wins on gene hold-out; causal wins on condition hold-out | The strongest scientific result: priors matter for distribution shift, not for interpolation. |
| TabPFN wins both | Tabular ICL exceeds specialised biology architectures on primary immune cells — a publishable negative result. |
**Grounding note for C3.** Cell-JEPA reports that, within a single cell line, JEPA improves absolute-state reconstruction but **not** effect-size (delta) estimation, and it did not test cross-context transfer. The condition hold-out is therefore the untested regime: a null result there is a corroboration, whereas a positive result there is a genuine extension. In either case, the 2×2 addresses a real question.
---
## 3. The split (immutable from Day 0)
Freeze `split_manifest.json` before any model sees the data; commit it; and record its SHA256 in the README. Every module verifies the SHA at startup.
- **Gene hold-out:** 15% of perturbed genes withheld entirely (an interpolation test, secondary).
- **Condition hold-out:** the full **Stim48hr** state held out; train on Rest + Stim8hr and test on Stim48hr (zero-shot; the **primary** test for C1–C3).
- **Donor probe:** Donor 4 reserved (a cross-donor sanity check).
`split_manifest.json` schema:
```json
{
  "seed": 42,
  "dataset": {"geo": "GSE278572", "doi": "<bioRxiv DOI>", "sha256_h5ad": "<hash>"},
  "hvg": {"n": 3000, "list_path": "data/split/hvg_3000.txt"},
  "gene_holdout": ["<ENSG...>", "..."],
  "condition_holdout": "Stim48hr",
  "donor_probe": "donor_4",
  "created_utc": "<iso>"
}
```
---
## 4. Repo layout (one repo)
```
tcell-perturb/
  core/                     # FROZEN shared foundation — build & commit FIRST (tag: core-frozen)
    data.py                 # backed h5ad reads, QC, HVG, subsampling
    pseudobulk.py           # (pert,cond,donor) mean profiles + deltas
    features.py             # ESM-2 + network/GO priors, DEG-frequency features
    split.py                # writes/loads split_manifest.json, verifies SHA
    eval.py                 # FROZEN metrics: pearson_delta_top50, perturbench_rank, DES, MAE...
    models/
      do_attention.py       # corrected DoAttention (§7d)
      gene_tokens.py         # CisTransCell-style token construction (§7d)
      causal_cistransformer.py
      jepa.py               # Cell-JEPA-style pretraining (§7e)
      baselines.py          # Ridge, TabPFN, PseudoBulk FCN
    voi.py                  # ensemble-disagreement VOI
    contract.py             # paths + schemas both worktrees code against (§5)
  runs/                     # each model writes results/<model>_<split>.parquet
  results/                  # benchmark_table.csv, benchmark_table_full.csv
  figures/
  hypotheses.md             # §2, committed before Day 1
  split_manifest.json       # immutable
  Snakefile
  environment.yml
  gpu_queue.py              # single-GPU job scheduler (§6)
```
---
## 5. Data contract (the interface — makes parallel work comparable)
Everything below is fixed in `core/contract.py`. Two worktrees, two people, or two Claude Code sessions all code against this interface and therefore never collide.
- **Split:** `split_manifest.json` together with its SHA256. All modules assert equality at startup.
- **Feature cache:** `data/embeddings/esm2.parquet` (gene→vector), `data/embeddings/context_prior.parquet` (gene→vector), `data/pseudobulk/{train,test}.parquet` (index = (pert_id, condition, donor); columns = HVG; plus a `delta` block), and `data/features/deg_freq.parquet` (a 50-dim BioMap feature).
- **Model output interface:** every model writes `runs/<model>_<split>.parquet`, with index = pert_id, columns = HVG, and values = the predicted **delta** (post − control). No model writes anywhere else.
- **Eval:** every model is scored by the *same* `core.eval.evaluate(pred_delta_df, split)` → a dict of metrics → appended to `results/benchmark_table.csv`. Metrics must not be reimplemented per model.
- **Namespacing:** model names are fixed strings: `ridge`, `tabpfn`, `fcn`, `causal`, `noncausal`, `jepa_only`, `jepa_causal`, `arc_state`. Filenames derive from these, so parallel writers never overwrite one another.
---
## 6. Concurrency model (single L4)
There are three lanes. Only **Lane G** touches the GPU, and it operates as a **serial queue**.
### Lane C — CPU (16 vCPU / 64 GB, always-on)
This lane runs continuously and uses no GPU: data download and QC, pseudobulk and deltas, DEG-frequency features, the network/GO prior, the split freeze, the eval harness, Ridge, TabPFN (CPU-feasible; GPU optional), figures, packaging, and **evaluation of any model the moment its `runs/*.parquet` lands**. This is the source of CPU/GPU overlap: while the L4 trains model *N*, Lane C scores model *N−1* and builds features for model *N+1*.
### Lane G — single L4 (serial job queue via `gpu_queue.py`)
One training or inference job runs at a time, in the following priority order (earlier positions protect a submittable result):
| # | Job | Rough L4 estimate | Fallback if over budget |
|---|---|---|---|
| G1 | ESM-2 650M embeddings (~8k genes) | <1 h | GenePT precomputed (instant) |
| G2 | CausalCisTransFormer (corrected mask) | 2–5 h* | 2-layer, d_model 128, gene-window 1000 |
| G3 | Non-causal twin (same arch) | 2–5 h* | same as G2 |
| G4 | **JEPA pretraining (single-cell, EMA)** | 8–12 h (overnight) | fewer cells (1M→0.5M), fewer steps, d_model 256→128 |
| G5 | JEPA+causal fine-tune + JEPA-only twin | 1–3 h | reduce epochs |
| G6 | Arc State ST train/inference | 2–4 h | Gate: mark N/A, drop from demo |
| G7 | If-time: GEARS, scVI, Geneformer/scGPT | remaining time only | skip |
\* The pseudobulk training sets are small (thousands of rows), so G2/G3/G5 are fast; **G4 is the only true bottleneck** and is the one component sized to a single overnight run.
**Measure-then-extrapolate gate (applies to every GPU job):** run **1 epoch / 200 steps**, log the wall-time, and extrapolate to the planned schedule. If the projected end-time exceeds the slot, apply that row's fallback immediately, rather than allowing a job to silently consume the queue.
### Lane D — development (optional, up to 2 Claude Code worktrees)
This provides concurrency in *writing code*, not in GPU use. The suggested split allows the two sessions to rarely block each other:
- **Worktree 1 ("core+causal"):** owns `core/` (data, pseudobulk, features, split, eval, contract), the baselines, `causal_cistransformer.py` and `do_attention.py`, and the GPU queue. It produces `core-frozen`.
- **Worktree 2 ("jepa+analysis"):** branches off `core-frozen`; owns `jepa.py`, the JEPA→causal integration, the 2×2 harness, `voi.py`, subsampling, and all figures.
Rule: Worktree 2 starts once `core-frozen` is tagged, since it imports the frozen contract. Both submit GPU jobs to the *single* `gpu_queue.py`, and they never launch training simultaneously. If a single session is used instead, follow the same order.
---
## 7. Component specs
### 7a. Data + QC + pseudobulk (Lane C)
- Use backed AnnData (`sc.read_h5ad(..., backed='r')`) or chunked reads; **never** materialise the full dense matrix (64 GB RAM). Subsample for JEPA (§7e).
- QC: apply standard cell and gene filters; normalise with log1p-CP10k; select 3,000 HVGs (scanpy); and persist the HVG list to the split.
- Pseudobulk: compute the mean profile per (pert_id, condition, donor); the delta = pert_pseudobulk − matched control_pseudobulk (same condition and donor). Write `data/pseudobulk/{train,test}.parquet`.
- DEG-frequency feature (the BioMap VCC-winner input): for the top-50 most-frequently-DE genes across the training perturbations, record the fraction of perturbations in which each is significant (BH FDR < 0.1). This is a 50-dim vector. Write `data/features/deg_freq.parquet`.
### 7b. Gene-token priors (Lane G G1, one-time)
- **Function/coding prior:** the ESM-2 650M embedding per gene's protein (mean-pooled). Cache to `esm2.parquet`. Fallback: GenePT (instant, precomputed on HF).
- **Regulatory/context prior:** the GenePT gene-text embedding **or** node2vec over a STRING/GRN graph (a gene→neighbourhood vector). Cache to `context_prior.parquet`. This inexpensively carries the "who regulates whom, and where does this gene sit in the network" signal.
- (Optional, spare time) DNABERT-2 promoter and Nucleotide Transformer coding embeddings.
### 7c. Baselines (Lane C, TabPFN optionally Lane G)
- **Ridge:** fit on the training pseudobulk deltas; predict all three splits. This is the canonical comparator.
- **TabPFN (v2 or v3):** per-target regression, with one TabPFN per top-50 DEG gene (target = that gene's delta), and features = control pseudobulk (PCA to ≤200 dims) ⊕ ESM-2(perturbed gene, PCA) ⊕ DEG-freq. Respect the feature ceiling (200 for v3 / 500 for v2) and the row ceiling (10k for v2 / 1M for v3). **Mirror the bioRxiv 2026.06.28.735106 protocol** so that this is a strong, rather than a strawman, competitor. It runs on CPU; use the GPU only if convenient.
- **PseudoBulk FCN:** a 3-layer FCN performing residual delta prediction on control pseudobulk ⊕ ESM-2 ⊕ DEG-freq (VCC 2nd-place-style). This takes approximately 1 h and can run on CPU or in a short GPU slot.
### 7d. CausalCisTransFormer (Lane G G2/G3)
Gene token (CisTransCell-style): `Z = phi_expr(expr) + phi_fuse(cat[phi_reg(context_prior), phi_cod(esm2)])`, together with **K=32 learned regulatory-proxy tokens** that mediate gene→gene effects (proxy round: `proxy = g2p(proxy, genes); genes = p2g(genes, proxy)`, run for 2 rounds, with the proxy update gated by each gene's regulator score).
**Corrected do-mask (the fix):**
```python
class DoAttention(nn.Module):
    """
    do(g_k = c): remove edges INTO the perturbed gene only (DoFormer, bioRxiv 2026.05.02.722054).
      - perturbed gene must NOT attend to others  -> mask its query row
      - other genes MUST still attend to it        -> DO NOT mask its key column (propagation!)
      - keep self-attention on k
      - set the perturbed gene's INPUT value to its knockdown level (CRISPRi != full KO)
    """
    def build_mask(self, L, perturbed_idx, device):
        M = torch.zeros(L, L, device=device)
        M[perturbed_idx, :] = float('-inf')      # cut incoming causal edges
        M[perturbed_idx, perturbed_idx] = 0.0    # keep self
        # NOTE: column perturbed_idx is intentionally NOT masked, so the
        # intervention propagates to downstream genes. (This is the deleted bug.)
        return M
```
- The non-causal twin is the identical architecture with the mask disabled (`use_causal_mask=False`). This is the single most important ablation; train it immediately after G2.
- Config: <30M params, BF16 autocast, gradient checkpointing, and random gene-window batching (1,000 of 3,000 HVG per forward). AdamW lr 2e-4, wd 1e-4, grad-clip 1.0, and a cosine schedule with 5% warmup. Loss = MSE_full + 0.5·MSE_delta + 0.1·BCE_DE. Early stop on the validation Pearson-delta (gene-hold-in val).
- Write the predicted delta to `runs/causal_<split>.parquet` and `runs/noncausal_<split>.parquet`.
### 7e. Cell-JEPA-style pretraining + integration (Lane G G4/G5)
**Correct recipe (single-cell resolution, EMA teacher):**
```python
# Cell-JEPA (arXiv 2602.02093): student(masked) predicts EMA-teacher(unmasked) cell embedding.
student   = GeneTokenEncoder(...)      # SAME class the causal model uses, so weights transfer
teacher   = copy.deepcopy(student)     # EMA; no gradient
for p in teacher.parameters(): p.requires_grad_(False)
predictor = MLP(d_model, d_model)
def jepa_step(cell):                    # cell = (gene_ids, values), <=600 HVG/cell (Cell-JEPA cap)
    masked = mask_values(cell, frac=0.5)          # mask EXPRESSION VALUES only, sentinel = -1
    e_s = student(masked).cls
    with torch.no_grad():
        e_t = teacher(cell).cls
    loss_jepa = 1.0 - cosine(predictor(e_s), e_t.detach())     # STOP-GRAD on teacher
    loss_rec  = mse(student.value_head(masked_positions), true_values)   # keep recon term
    return w_jepa*loss_jepa + w_rec*loss_rec       # e.g. w_jepa=1.0, w_rec=0.5
# after each optimizer.step():
ema_update(teacher, student, m=0.996)   # momentum; ramp 0.996 -> 0.999
```
- **Single-cell, subsampled.** Draw approximately 1–2M cells from the 22M (stratified over donor × condition), with ≤600 HVG per cell. Do **not** use a pseudobulk MLP, which discards the reason to use JEPA and cannot initialise a per-gene transformer.
- **L4 sizing:** attention is 600² per cell (very small), so batches of cells fit comfortably in 24 GB at BF16. The bottleneck is the number of steps × cells, not memory — hence the overnight slot and the "fewer cells / fewer steps" fallback.
- **Integration.** Because `student` is the *same* `GeneTokenEncoder` class used inside `CausalCisTransFormer`, initialise the causal model's encoder from the JEPA checkpoint and then fine-tune on the pseudobulk delta task. This yields the two JEPA cells of the 2×2 (`jepa_causal` and `jepa_only`). The random-init cells are simply G2/G3.
- **Collapse guard.** Log the standard deviation of the teacher embeddings per batch; if it trends toward 0, the run is collapsing, and the response is to increase predictor capacity or EMA momentum, or to add a small variance-covariance (VICReg-style) penalty. This should not occur with the recipe above, but it should nonetheless be monitored.
### 7f. 2×2 ablation harness (Lane D worktree 2)
The four models (`noncausal`, `causal`, `jepa_only`, `jepa_causal`) are evaluated on the frozen splits via `core.eval`. Produce the 2×2 bar chart on **condition-hold-out Pearson-delta** (Figure 2). The causal-vs-noncausal contrast (C2) and the JEPA-init contrast (C3) both read directly off this chart.
### 7g. Arc State (Lane G G6, gated)
- Train an ST model on the CD4+ **train** split (State's `state tx`), and infer on the test splits. This is transfer, not pure zero-shot, and should be stated as such.
- **Gate 3:** if the perturbation/format API does not run cleanly within its slot, mark `arc_state = "N/A — resource limit"` and drop it from the demo. Do not allow it to block packaging.
### 7h. VOI + subsampling (Lane D worktree 2, Lane G small)
- VOI is the mean pairwise L2 disagreement across the ensemble's per-gene predictions (with no normalising-flow dependency). High disagreement indicates high epistemic uncertainty, which is most worth measuring.
- Subsampling curve: train the ensemble on 5/10/20/50/100% of the training perturbations; evaluate condition-hold-out Pearson-delta at each fraction; use 3 random replicates plus 1 VOI-guided selection per fraction. Annotate where 90% of full-screen accuracy is reached (Figure 3).
- (Spare time) PRESCRIBE calibrated uncertainty (arXiv) as an upgrade to disagreement-VOI; Gate: abandon it if the flow does not converge within approximately 2 h.
### 7i. Eval harness + mode-collapse detector (core/eval.py, frozen)
- Headline metrics (demo): **Pearson-delta on the top-50 DEGs** (accuracy), **PerturBench rank / perturbation-discrimination** (a mode-collapse detector — flag any model > 0.4 in red), and **DES** (sign-correct DEG overlap). Report these separately for the gene hold-out and the condition hold-out.
- Full battery (repository appendix, `benchmark_table_full.csv`): additionally MAE, Spearman LFC, Spearman effect-size, AUPRC, and E-distance. This runs automatically and is never shown in the demo.
- The mode-collapse detector is essential, because many models "win" on MAE by predicting the control or the mean. This must be surfaced explicitly.
---
## 8. L4 compute + memory playbook
- **Always use BF16 autocast** (Ada tensor cores). Apply gradient checkpointing on all transformer blocks.
- For the causal model over G=3000, gene-window batching (1,000/forward) keeps attention memory bounded.
- For JEPA, ≤600 genes/cell makes memory a non-issue; throughput (steps) is the constraint.
- For data, use backed/chunked h5ad and subsample for JEPA; the pseudobulk fits in RAM easily.
- For TabPFN, the per-gene models are small and CPU is sufficient; cap features at 200 (v3) / 500 (v2).
- **The total GPU budget on a single L4 for a submittable and novel result is roughly one overnight run (JEPA) plus approximately 1 working day of short jobs (everything else).** The measure-then-extrapolate gate (§6) is what keeps this estimate honest.
---
## 9. Day-by-day (compressed, parallel lanes marked)
**Day 0 (evening) — freeze the core.** Lane C: download, QC, pseudobulk, DEG-freq, split freeze, and `hypotheses.md` committed. Lane G: G1 (ESM-2 embeddings). Tag `core-frozen`. *Milestone: contract and features frozen; the second worktree can start.*
**Day 1 — baselines + causal, submittable by end of day.** Lane C: Ridge, TabPFN, and FCN evaluated. Lane G: G2 (causal) → G3 (non-causal). Lane C scores each as it lands. *Milestone: C1 and C2 testable on the frozen splits → the **first submittable result**.*
**Day 2 — JEPA overnight.** Lane G: initiate G4 (JEPA pretraining) to run overnight, measuring epoch 1 first. Lanes C/D: build the 2×2 harness, VOI, and eval figures against the models already completed.
**Day 3 — complete the 2×2 + Arc State.** Lane G: G5 (JEPA+causal fine-tune + JEPA-only) → G6 (Arc State, gated). Lane C: full benchmark table, 2×2 chart, and VOI/subsampling curve. *Milestone: the **project is fully submittable**, with all three claims answered.*
**Day 4 — additive only.** Lane G: G7 spare-time models as time permits. Lanes C/D: the PRESCRIBE upgrade (gated), and the biology annotation figure (top-20 VOI-disagreement genes × gene family × T-cell role × GWAS flag, approximately 30 min).
**Day 5 — package + demo.** Snakefile end-to-end; pinned `environment.yml`; README with the dataset DOI, split SHA, architecture diagram, and one-command rerun; tag `v1.0-hackathon`. Rehearse the 5-minute demo (§12).
---
## 10. Gates + fallbacks
| Gate | Trigger | Action |
|---|---|---|
| Every GPU job | epoch-1 wall-time extrapolates past slot | apply that job's §6 fallback immediately |
| Causal training | val loss diverges after epoch 5 | ESM-2-only tokens, 2-layer, d_model 128 |
| JEPA | teacher-embedding std → 0 (collapse) | ↑ EMA momentum / predictor capacity; add VICReg penalty |
| JEPA | over budget | 1M→0.5M cells, fewer steps, smaller d_model |
| Arc State | API/format won't run in slot | mark N/A, drop from demo |
| PRESCRIBE | flow not converged in ~2 h | keep disagreement-VOI |
---
## 11. Acceptance criteria (submittable checkpoints)
- **CP1 (Day 1):** `results/benchmark_table.csv` contains Ridge, TabPFN, FCN, causal, and non-causal on the gene and condition hold-outs; the split SHA is verified; and the eval harness is passing. *Submittable.*
- **CP2 (Day 3):** the 2×2 is complete (`jepa_only` and `jepa_causal` added); VOI scores and the subsampling curve are present; three demo figures exist; and `snakemake --cores all` runs end-to-end. *Fully submittable.*
- **CP3 (Day 5):** the reproducibility package is tagged; the full 8-metric appendix is present; and the README supports a one-command rerun.
---
## 12. Demo (3 figures, one sentence, 5 minutes)
The sentence for the judges to remember:
> "Standard models treat a gene knockdown as an observation. We treat it as an intervention — and that distinction is what lets the model predict a knockdown's effect in an activation state it has never seen."
- **Figure 1 — benchmark table.** Pearson-delta, PerturBench rank (red = collapsed), and DES; with two highlighted rows (the best on each hold-out). The full 8-metric table appears in the appendix.
- **Figure 2 — 2×2 ablation** on condition-hold-out Pearson-delta. This reads out both the do-operator effect (C2) and the JEPA-init effect (C3) in a single chart.
- **Figure 3 — sample-efficiency curve.** Random versus VOI-guided subsampling; annotate the 90%-of-full-screen point.
Narrative beats: the dataset (30s) → the intervention-versus-observation failure mode and the corrected mask (60s) → the benchmark and collapse detector (60s) → the 2×2 causal/JEPA result against the pre-registered hypothesis (90s) → the VOI experimental payoff (60s).
---
## 13. Claude Code kickoff prompts
**Single-session version:**
> Build the project in `UNIFIED_BUILD_PLAN.md`. Start with §4–5: scaffold the repo and `core/contract.py`, then build and commit `core/` (data, pseudobulk, features, split, eval) and tag `core-frozen`. Freeze `split_manifest.json` and commit `hypotheses.md` (§2) before training anything. Implement the corrected `DoAttention` exactly as in §7d and the JEPA recipe exactly as in §7e — do not add `M[:, perturbed]=-inf`, and do not drop the EMA teacher. Schedule all GPU work through `gpu_queue.py` in the §6 order, running the epoch-1 measure-then-extrapolate gate before each job. Target CP1 (§11) by end of Day 1.
**Two-worktree version — Worktree 1 (core+causal):**
> Own `core/`, baselines, and the causal model per `UNIFIED_BUILD_PLAN.md` §4–7d and §5 (contract). Build and tag `core-frozen` first. Implement the corrected `DoAttention` (§7d). Own `gpu_queue.py`. Deliver CP1.
**Two-worktree version — Worktree 2 (jepa+analysis):**
> Branch off `core-frozen`. Own `jepa.py` (§7e, EMA teacher + single-cell masking — the recipe is exact, follow it), the JEPA→causal integration, the 2×2 harness (§7f), `voi.py` + subsampling (§7h), and all figures. Submit JEPA jobs to the shared `gpu_queue.py`; never launch training while Worktree 1 has a GPU job running. Deliver the 2×2 chart + VOI curve for CP2.
