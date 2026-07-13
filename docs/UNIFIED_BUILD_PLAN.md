# CD4+ T Cell Perturbation Prediction — Unified Build Plan
**One repo, one pre-registration, three claims, run on a single NVIDIA L4 (24 GB) with parallel CPU/GPU lanes.**
Dataset: Marson/Pritchard genome-scale CRISPRi Perturb-seq in primary human CD4+ T cells (~22M cells, every expressed gene silenced one at a time, 4 donors, 3 activation states: Rest / Stim8hr / Stim48hr). GEO GSE278572; CZI Virtual Cells mirror.
This plan merges the former "v3 (causal + VOI)" and "v4 (JEPA + causal)" plans. They overlapped ~80%, so they are now a single pipeline whose experimental core is a **2×2 ablation (JEPA-init × causal-mask)**. The causal claim, the do-operator isolation, and the JEPA claim are three cells/rows of that one matrix. TabPFN / Ridge / PseudoBulk-FCN / Arc State are external reference points; VOI is the applied layer.
---
## 0. How to use this document
This is written to drive **one or two Claude Code sessions**. If you use two, run them as `git worktree` checkouts of the same repo (see §5). The build is structured so a submittable result exists early and everything after is additive.
Kickoff prompts are in §13. Read §2 (claims), §3 (corrections — do not re-introduce these bugs), §4–5 (layout + contract), and §6 (concurrency) before writing any code.
---
## 1. Corrections carried in from the plan review (authoritative — do not revert)
Both prior plans contained two errors that would silently break the centrepiece. These are fixed below and must stay fixed.
1. **Causal do-mask over-masks (do-calculus error).** Per DoFormer (Karbalayghareh et al., bioRxiv 2026.05.02.722054), an intervention removes only edges *into* the perturbed gene: the perturbed gene stops attending to others, but **others must still attend to it** so the intervention propagates downstream. The prior plans added `M[:, perturbed] = -inf`, which severs outgoing edges and deletes the signal being predicted. **Delete that line.** See §7d for the corrected `DoAttention`.
2. **JEPA recipe collapses as written.** Per Cell-JEPA (arXiv 2602.02093), JEPA needs a **student on masked input + EMA teacher on unmasked input (stop-gradient) + predictor head + cosine loss**, and it operates at **single-cell** resolution (mask expression *values* within a cell), not on a pseudobulk MLP. The prior "no EMA, plain L2, pseudobulk" is a collapse path and discards the 22M cells. See §7e.
Additional corrections:
- **TabPFN version.** Use TabPFN v2 (≤500 features, ≤10k rows) or TabPFN-3 (≤1M rows, ≤200 features). The prior "1024 samples / 50 dims" is v1. Mirror the protocol of the Tabular-Foundation-Models perturbation paper (bioRxiv 2026.06.28.735106), which already ran TabPFN on a genome-wide CRISPR screen in primary human CD4+ T cells and found tabular models beat specialized ones on pseudobulk. Your novel contribution is therefore specifically the **condition hold-out + causal structure**, not pseudobulk accuracy in general.
- **Arc State.** State is validated for *context transfer*, not zero-shot unseen perturbations within a cell line (STRAND, arXiv 2602.10156). You will likely have to train an ST model on your CD4+ train split (transfer, not pure zero-shot). Keep the Gate that drops State to "N/A — resource limit" if it will not run cleanly.
- **Gene-token priors.** Prefer ESM-2 (function/coding prior) + a network/GO prior (regulatory/context prior, e.g. GenePT text embedding or a STRING node2vec vector) over DNABERT-2 + Nucleotide Transformer. This is lighter on one L4 and directly addresses "represent a gene never silenced" via function + network neighborhood. DNABERT-2/NT are optional If-Time.
- **Stacking.** Do not K-fold-stack the transformer (too expensive on one GPU). Stack only cheap base models (Ridge / TabPFN / FCN) with out-of-fold predictions; treat the transformer as a standalone comparator.
- **Demo claim hygiene.** Drop "6 months computational exclusivity." Say "recently released (Dec 2025); among the first models built on it."
---
## 2. Pre-registration (write `hypotheses.md` and commit BEFORE Day 1)
**Claims**
- **C1 (causal, external).** CausalCisTransFormer with the corrected do-mask matches or beats strong baselines **including TabPFN** on the **condition hold-out** (zero-shot Stim48hr).
- **C2 (do-operator isolation).** Causal mask beats its non-causal twin on the condition hold-out (2×2 row contrast). Report regardless of leaderboard position.
- **C3 (JEPA).** JEPA-init helps the condition hold-out; full 2×2 (JEPA × causal).
- **S1 (support / VOI).** Model-disagreement ranks which perturbations are most worth measuring; VOI-guided selection reaches ~90% of full-screen accuracy from a fraction of perturbations.
**The 2×2 (experimental core).** Encoder-init × causal-mask:
| Encoder init | Causal mask | Label |
|---|---|---|
| Random | off | Direct-regression baseline |
| Random | on | Causal-only (= C2 treatment) |
| JEPA | off | JEPA-only |
| JEPA | on | **JEPA + causal (main model)** |
**Pre-computed outcome interpretations (written now, so Day 5 is analysis not scrambling):**
| Outcome | Interpretation |
|---|---|
| Causal beats non-causal on condition hold-out (any init) | do-operator provides real inductive bias under activation-state shift. Headline. |
| JEPA-init beats random-init on condition hold-out | representation robustness transfers to cross-context perturbation delta — **extends Cell-JEPA**, which only tested within a single cell line. Report prominently. |
| JEPA helps absolute-state but not gene-hold-out delta | **corroborates Cell-JEPA's "complementary aspects" finding.** Clean positive-or-null, still publishable. |
| TabPFN wins on gene hold-out; causal wins on condition hold-out | the strongest scientific result: priors matter for distribution shift, not interpolation. |
| TabPFN wins both | tabular ICL beats specialized biology architectures on primary immune cells — publishable negative. |
**Grounding note for C3.** Cell-JEPA reports that within a single cell line, JEPA improves absolute-state reconstruction but **not** effect-size (delta) estimation, and did not test cross-context transfer. So the condition hold-out is the untested regime; a null there is a corroboration, a positive there is a genuine extension. Either way the 2×2 answers a real question.
---
## 3. The split (immutable from Day 0)
Freeze `split_manifest.json` before any model sees data; commit; record SHA256 in the README. Every module verifies the SHA at startup.
- **Gene hold-out:** 15% of perturbed genes withheld entirely (interpolation test, secondary).
- **Condition hold-out:** full **Stim48hr** held out; train on Rest + Stim8hr, test on Stim48hr (zero-shot; **primary** test for C1–C3).
- **Donor probe:** Donor 4 reserved (cross-donor sanity check).
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
Everything below is fixed in `core/contract.py`. Two worktrees, or two people, or two Claude Code sessions, all code against this and never collide.
- **Split:** `split_manifest.json` + its SHA256. All modules assert equality at startup.
- **Feature cache:** `data/embeddings/esm2.parquet` (gene→vector), `data/embeddings/context_prior.parquet` (gene→vector), `data/pseudobulk/{train,test}.parquet` (index = (pert_id, condition, donor); columns = HVG; plus a `delta` block), `data/features/deg_freq.parquet` (50-dim BioMap feature).
- **Model output interface:** every model writes `runs/<model>_<split>.parquet`, index = pert_id, columns = HVG, values = predicted **delta** (post − control). No model writes anywhere else.
- **Eval:** every model is scored by the *same* `core.eval.evaluate(pred_delta_df, split)` → dict of metrics → appended to `results/benchmark_table.csv`. Do not reimplement metrics per model.
- **Namespacing:** model names are fixed strings: `ridge`, `tabpfn`, `fcn`, `causal`, `noncausal`, `jepa_only`, `jepa_causal`, `arc_state`. Filenames derive from these, so parallel writers never overwrite each other.
---
## 6. Concurrency model (single L4)
Three lanes. Only **Lane G** touches the GPU, and it is a **serial queue**.
### Lane C — CPU (16 vCPU / 64 GB, always-on)
Runs continuously, no GPU: data download + QC, pseudobulk + deltas, DEG-frequency features, network/GO prior, split freeze, eval harness, Ridge, TabPFN (CPU-feasible; GPU optional), figures, packaging, and **evaluation of any model the moment its `runs/*.parquet` lands**. This is where CPU/GPU overlap comes from: while the L4 trains model *N*, Lane C scores model *N−1* and builds features for model *N+1*.
### Lane G — single L4 (serial job queue via `gpu_queue.py`)
One training/inference job at a time, in this priority order (earlier = protects a submittable result):
| # | Job | Rough L4 estimate | Fallback if over budget |
|---|---|---|---|
| G1 | ESM-2 650M embeddings (~8k genes) | <1 h | GenePT precomputed (instant) |
| G2 | CausalCisTransFormer (corrected mask) | 2–5 h* | 2-layer, d_model 128, gene-window 1000 |
| G3 | Non-causal twin (same arch) | 2–5 h* | same as G2 |
| G4 | **JEPA pretraining (single-cell, EMA)** | 8–12 h (overnight) | fewer cells (1M→0.5M), fewer steps, d_model 256→128 |
| G5 | JEPA+causal fine-tune + JEPA-only twin | 1–3 h | reduce epochs |
| G6 | Arc State ST train/inference | 2–4 h | Gate: mark N/A, drop from demo |
| G7 | If-time: GEARS, scVI, Geneformer/scGPT | remaining time only | skip |
\* Pseudobulk training sets are small (thousands of rows), so G2/G3/G5 are fast; **G4 is the only true bottleneck** and is the one component sized to a single overnight run.
**Measure-then-extrapolate gate (applies to every GPU job):** run **1 epoch / 200 steps**, log wall-time, extrapolate to the planned schedule. If projected end-time exceeds the slot, apply that row's fallback immediately — do not let a job silently eat the queue.
### Lane D — development (optional, up to 2 Claude Code worktrees)
Concurrency in *writing code*, not in GPU use. Suggested split so the two sessions rarely block each other:
- **Worktree 1 ("core+causal"):** owns `core/` (data, pseudobulk, features, split, eval, contract), baselines, `causal_cistransformer.py` + `do_attention.py`, and the GPU queue. Produces `core-frozen`.
- **Worktree 2 ("jepa+analysis"):** branches off `core-frozen`; owns `jepa.py`, JEPA→causal integration, the 2×2 harness, `voi.py`, subsampling, and all figures.
Rule: Worktree 2 starts once `core-frozen` is tagged (it imports the frozen contract). Both submit GPU jobs to the *single* `gpu_queue.py`; they never launch training simultaneously. If you use one session instead, just follow the same order.
---
## 7. Component specs
### 7a. Data + QC + pseudobulk (Lane C)
- Backed AnnData (`sc.read_h5ad(..., backed='r')`) or chunked reads; **never** materialize the full dense matrix (64 GB RAM). Subsample for JEPA (§7e).
- QC: standard cell/gene filters; normalize log1p-CP10k; select 3,000 HVGs (scanpy); persist HVG list to the split.
- Pseudobulk: mean profile per (pert_id, condition, donor); delta = pert_pseudobulk − matched control_pseudobulk (same condition, donor). Write `data/pseudobulk/{train,test}.parquet`.
- DEG-frequency feature (BioMap VCC-winner input): for the top-50 most-frequently-DE genes across training perturbations, the fraction of perturbations where each is significant (BH FDR < 0.1). 50-dim vector. Write `data/features/deg_freq.parquet`.
### 7b. Gene-token priors (Lane G G1, one-time)
- **Function/coding prior:** ESM-2 650M embedding per gene's protein (mean-pooled). Cache to `esm2.parquet`. Fallback: GenePT (instant, precomputed on HF).
- **Regulatory/context prior:** GenePT gene-text embedding **or** node2vec over a STRING/GRN graph (gene→neighborhood vector). Cache to `context_prior.parquet`. This carries the "who regulates whom / where does this gene sit in the network" signal cheaply.
- (Optional If-Time) DNABERT-2 promoter + Nucleotide Transformer coding embeddings.
### 7c. Baselines (Lane C, TabPFN optionally Lane G)
- **Ridge:** fit on training pseudobulk deltas; predict all three splits. Canonical comparator.
- **TabPFN (v2 or v3):** per-target regression — one TabPFN per top-50 DEG gene (target = that gene's delta), features = control pseudobulk (PCA to ≤200 dims) ⊕ ESM-2(perturbed gene, PCA) ⊕ DEG-freq. Respect feature ceiling (200 for v3 / 500 for v2) and row ceiling (10k v2 / 1M v3). **Mirror the bioRxiv 2026.06.28.735106 protocol** so this is a strong, not strawman, competitor. Runs on CPU; use GPU only if convenient.
- **PseudoBulk FCN:** 3-layer FCN, residual delta prediction on control pseudobulk ⊕ ESM-2 ⊕ DEG-freq (VCC 2nd-place-style). ~1 h; can run CPU or a short GPU slot.
### 7d. CausalCisTransFormer (Lane G G2/G3)
Gene token (CisTransCell-style): `Z = phi_expr(expr) + phi_fuse(cat[phi_reg(context_prior), phi_cod(esm2)])`, plus **K=32 learned regulatory-proxy tokens** that mediate gene→gene effects (proxy round: `proxy = g2p(proxy, genes); genes = p2g(genes, proxy)`, 2 rounds, proxy update gated by each gene's regulator score).
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
- Non-causal twin = identical architecture with the mask disabled (`use_causal_mask=False`). This is the single most important ablation; train it right after G2.
- Config: <30M params, BF16 autocast, gradient checkpointing, random gene-window batching (1,000 of 3,000 HVG per forward). AdamW lr 2e-4, wd 1e-4, grad-clip 1.0, cosine schedule w/ 5% warmup. Loss = MSE_full + 0.5·MSE_delta + 0.1·BCE_DE. Early stop on val Pearson-delta (gene-hold-in val).
- Output predicted delta → `runs/causal_<split>.parquet`, `runs/noncausal_<split>.parquet`.
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
- **Single-cell, subsampled.** Draw ~1–2M cells from the 22M (stratified over donor × condition), ≤600 HVG per cell. Do **not** use a pseudobulk MLP — that discards the reason to use JEPA and cannot initialize a per-gene transformer.
- **L4 sizing:** attention is 600² per cell (tiny), so batches of cells fit comfortably in 24 GB at BF16. Bottleneck is number of steps × cells, not memory — hence the overnight slot and the "fewer cells / fewer steps" fallback.
- **Integration.** Because `student` is the *same* `GeneTokenEncoder` class used inside `CausalCisTransFormer`, initialize the causal model's encoder from the JEPA checkpoint, then fine-tune on the pseudobulk delta task. This yields the two JEPA cells of the 2×2 (`jepa_causal`, `jepa_only`). The random-init cells are just G2/G3.
- **Collapse guard.** Log the std of teacher embeddings per batch; if it trends toward 0, the run is collapsing — increase predictor capacity / EMA momentum, or add a small variance-covariance (VICReg-style) penalty. This should not happen with the recipe above, but monitor it.
### 7f. 2×2 ablation harness (Lane D worktree 2)
Four models (`noncausal`, `causal`, `jepa_only`, `jepa_causal`) evaluated on the frozen splits via `core.eval`. Produce the 2×2 bar chart on **condition-hold-out Pearson-delta** (Figure 2). The causal-vs-noncausal contrast (C2) and the JEPA-init contrast (C3) both read directly off this chart.
### 7g. Arc State (Lane G G6, gated)
- Train an ST model on the CD4+ **train** split (State's `state tx`), infer on test splits. This is transfer, not pure zero-shot — state that honestly.
- **Gate 3:** if the perturbation/format API does not run cleanly within its slot, mark `arc_state = "N/A — resource limit"` and drop it from the demo. Do not let it block packaging.
### 7h. VOI + subsampling (Lane D worktree 2, Lane G small)
- VOI = mean pairwise L2 disagreement across the ensemble's per-gene predictions (no normalizing-flow dependency). High disagreement = high epistemic uncertainty = most worth measuring.
- Subsampling curve: train the ensemble on 5/10/20/50/100% of training perturbations; evaluate condition-hold-out Pearson-delta at each fraction; 3 random replicates + 1 VOI-guided selection per fraction. Annotate where 90% of full-screen accuracy is reached (Figure 3).
- (If-Time) PRESCRIBE calibrated uncertainty (arXiv) as an upgrade to disagreement-VOI; Gate: abandon if the flow does not converge in ~2 h.
### 7i. Eval harness + mode-collapse detector (core/eval.py, frozen)
- Headline metrics (demo): **Pearson-delta on top-50 DEGs** (accuracy), **PerturBench rank / perturbation-discrimination** (mode-collapse detector — flag any model > 0.4 in red), **DES** (sign-correct DEG overlap). Report separately for gene hold-out and condition hold-out.
- Full battery (repo appendix, `benchmark_table_full.csv`): + MAE, Spearman LFC, Spearman effect-size, AUPRC, E-distance. Runs automatically; never shown in demo.
- The mode-collapse detector is essential: many models "win" MAE by predicting the control/mean. Surface that explicitly.
---
## 8. L4 compute + memory playbook
- **Always BF16 autocast** (Ada tensor cores). Gradient checkpointing on all transformer blocks.
- Causal model over G=3000: gene-window batching (1,000/forward) keeps attention memory bounded.
- JEPA: ≤600 genes/cell → memory is a non-issue; throughput (steps) is the constraint.
- Data: backed/chunked h5ad; subsample for JEPA; pseudobulk fits in RAM easily.
- TabPFN: per-gene models are small; CPU is fine; cap features at 200 (v3) / 500 (v2).
- **Total GPU budget on one L4 for a submittable + novel result: roughly one overnight (JEPA) + ~1 working day of short jobs (everything else).** The measure-then-extrapolate gate (§6) is what keeps this honest.
---
## 9. Day-by-day (compressed, parallel lanes marked)
**Day 0 (evening) — freeze the core.** Lane C: download + QC + pseudobulk + DEG-freq + split freeze + `hypotheses.md` committed. Lane G: G1 (ESM-2 embeddings). Tag `core-frozen`. *Milestone: contract + features frozen; second worktree can start.*
**Day 1 — baselines + causal, submittable by EOD.** Lane C: Ridge + TabPFN + FCN evaluated. Lane G: G2 (causal) → G3 (non-causal). Lane C scores each as it lands. *Milestone: C1 + C2 testable on the frozen splits → **first submittable result**.*
**Day 2 — JEPA overnight.** Lane G: kick off G4 (JEPA pretraining) to run overnight; measure epoch 1 first. Lane C/D: build the 2×2 harness + VOI + eval figures against the models already done.
**Day 3 — complete the 2×2 + Arc State.** Lane G: G5 (JEPA+causal fine-tune + JEPA-only) → G6 (Arc State, gated). Lane C: full benchmark table + 2×2 chart + VOI/subsampling curve. *Milestone: **project fully submittable**, all three claims answered.*
**Day 4 — additive only.** Lane G: G7 if-time models as time permits. Lane C/D: PRESCRIBE upgrade (gated), biology annotation figure (top-20 VOI-disagreement genes × gene family × T-cell role × GWAS flag, ~30 min).
**Day 5 — package + demo.** Snakefile end-to-end; pinned `environment.yml`; README with dataset DOI + split SHA + architecture diagram + one-command rerun; tag `v1.0-hackathon`. Rehearse the 5-minute demo (§12).
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
- **CP1 (Day 1):** `results/benchmark_table.csv` has Ridge, TabPFN, FCN, causal, non-causal on gene + condition hold-out; split SHA verified; eval harness passing. *Submittable.*
- **CP2 (Day 3):** 2×2 complete (`jepa_only`, `jepa_causal` added); VOI scores + subsampling curve; three demo figures; `snakemake --cores all` runs end-to-end. *Fully submittable.*
- **CP3 (Day 5):** reproducibility package tagged; full 8-metric appendix; README one-command rerun.
---
## 12. Demo (3 figures, one sentence, 5 minutes)
The sentence judges remember:
> "Standard models treat a gene knockdown as an observation. We treat it as an intervention — and that distinction is what lets the model predict a knockdown's effect in an activation state it has never seen."
- **Figure 1 — benchmark table.** Pearson-delta, PerturBench rank (red = collapsed), DES; two highlighted rows (best on each hold-out). Full 8-metric table in the appendix.
- **Figure 2 — 2×2 ablation** on condition-hold-out Pearson-delta. Reads out both the do-operator effect (C2) and the JEPA-init effect (C3) in one chart.
- **Figure 3 — sample-efficiency curve.** Random vs VOI-guided subsampling; annotate 90%-of-full-screen point.
Narrative beats: dataset (30s) → the intervention-vs-observation failure mode + the corrected mask (60s) → benchmark + collapse detector (60s) → the 2×2 causal/JEPA result vs pre-registered hypothesis (90s) → VOI experimental payoff (60s).
---
## 13. Claude Code kickoff prompts
**Single-session version:**
> Build the project in `UNIFIED_BUILD_PLAN.md`. Start with §4–5: scaffold the repo and `core/contract.py`, then build and commit `core/` (data, pseudobulk, features, split, eval) and tag `core-frozen`. Freeze `split_manifest.json` and commit `hypotheses.md` (§2) before training anything. Implement the corrected `DoAttention` exactly as in §7d and the JEPA recipe exactly as in §7e — do not add `M[:, perturbed]=-inf`, and do not drop the EMA teacher. Schedule all GPU work through `gpu_queue.py` in the §6 order, running the epoch-1 measure-then-extrapolate gate before each job. Target CP1 (§11) by end of Day 1.
**Two-worktree version — Worktree 1 (core+causal):**
> Own `core/`, baselines, and the causal model per `UNIFIED_BUILD_PLAN.md` §4–7d and §5 (contract). Build and tag `core-frozen` first. Implement the corrected `DoAttention` (§7d). Own `gpu_queue.py`. Deliver CP1.
**Two-worktree version — Worktree 2 (jepa+analysis):**
> Branch off `core-frozen`. Own `jepa.py` (§7e, EMA teacher + single-cell masking — the recipe is exact, follow it), the JEPA→causal integration, the 2×2 harness (§7f), `voi.py` + subsampling (§7h), and all figures. Submit JEPA jobs to the shared `gpu_queue.py`; never launch training while Worktree 1 has a GPU job running. Deliver the 2×2 chart + VOI curve for CP2.
