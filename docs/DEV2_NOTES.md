# Developer 2 — JEPA + analysis half (status + handshake)

Branch `dev2` (worktree `../cd4-ws2`). **Reconciled to Developer 1's real core-frozen
code** (merged origin/main). 61 tests green (my 51 + Dev1's 10); JEPA verified to learn
on Dev1's real `GeneTokenEncoder`; a JEPA checkpoint loads into Dev1's real
`CausalCisTransFormer.encoder`.

## What's here

| File | Role | Plan |
|---|---|---|
| `core/models/jepa.py` | Cell-JEPA: masked-value student, stop-grad EMA teacher, predictor, cosine + recon loss, VICReg collapse guard, measure-then-extrapolate gate, checkpointing. Entry points `run_jepa` (G4) / `run_jepa_finetune` (G5) | §7e |
| `core/models/jepa_data.py` | Windowed single-cell feed `(values, esm2, ctx)` + **mmap-able `.npy`** cell cache + `materialize_cell_cache` (Task 2) | §7e/§2 |
| `core/models/jepa_integration.py` | JEPA → causal weight transfer + G5 fine-tune | §7f |
| `core/voi.py` | Ensemble-disagreement VOI + subsampling curve | §7h |
| `core/ablation.py` | 2×2 harness + C2/C3 contrasts | §7f |
| `figures/make_figures.py` | Figures 1–4 | §12 |
| `tests/` | 51 dev2 tests (JEPA numerics, weight transfer into the real causal model, VOI, ablation, figures) | — |

Run: `.venv/bin/python -m pytest tests/ -q`.

## How I consume Developer 1's real interfaces (verified)
- **Encoder** `core.models.gene_tokens.GeneTokenEncoder(d_model, esm2_dim, ctx_dim, n_proxy, n_proxy_rounds, n_heads)`,
  `forward(values [B,L], esm2 [L,esm2_dim], ctx [L,ctx_dim]) -> EncoderOutput(tokens, cls)`, has `value_head`.
  My JEPA uses it as the student; `student.state_dict()` loads into `causal_model.encoder` key-for-key
  (`test_transfer_into_real_causal_model`). JEPAConfig defaults (`d_model`, `esm2_dim=ESM2_DIM`,
  `ctx_dim=CONTEXT_PRIOR_DIM`, `n_proxy=32`, `n_proxy_rounds=2`, `n_heads=4`) match `CausalConfig` so the transfer is exact.
- **Feed**: single-cell resolution via a shared gene **window** per batch (≤600 HVG, §7e) — the same
  gene-window batching the causal model uses — so `esm2/ctx` are `[W,dim]` shared, matching the encoder.
- **Sampler**: I materialize `CELLS_DIR` with Dev1's `data.stratified_cell_indices` + `read_backed`
  (Dev1's `data.py` has the index sampler but no `CELLS_DIR` writer — that writer is mine).
- **Eval**: `core.eval.evaluate(pred_df, split) -> dict` returns the contract metric keys; my ablation consumes it as-is.
- **Queue**: `gpu_queue.py` dispatches `jepa → jepa.run_jepa`, `jepa_finetune → jepa.run_jepa_finetune` (both exist).

## 🤝 Handshake items for Developer 1 (flagged, not blocking)
1. **G5 encoder-init hook (preferred).** To keep the 2×2 "same trainer, only init differs," add an
   `encoder_init_ckpt=None` param to `causal_cistransformer._run` that calls
   `load_jepa_into_encoder(model.encoder, encoder_init_ckpt)` right after `_build_model`.
   `finetune_jepa_models` auto-detects and uses it; **until then it runs a faithful replica of `_run`
   with the init inserted** (depends on your `_build_model`/`_train`/`_predict_split` privates — kept in lockstep).
2. **Benchmark writes.** You write rows via `eval.evaluate_and_record`; my `ablation.upsert_benchmark`
   also upserts (keep='last', no clobber). Let's pick one writer for the JEPA rows to avoid double entries.
3. **Cell-cache helper (nice-to-have).** `materialize_cell_cache` uses `data.cells_to_hvg_matrix(adata, idx, hvg)`
   if you expose it (log1p-CP10k over the HVG panel, chunked); otherwise it falls back to a local chunked densify.

## 🚦 GPU / box status — NOT yet time to run G4
Box `ssh ubuntu@54.163.21.62`: clean **L4 (23 GB), idle, 0 procs, no repo, no `~/cd4-perturb-data`, no cell cache**.
Nothing deployed or run there yet. My G4 (JEPA pretraining) is the Day-2 **overnight** job and its prerequisites
are not met:
- [ ] `esm2`/`context_prior` caches built (G1) — Dev1's `esm2` job needs a gene→sequence map first.
- [ ] cell cache materialized — **needs the raw 22M-cell GSE278572 `.h5ad` (a >5 GB download → your approval).**
- [ ] Dev1's G2/G3 (`causal`/`noncausal`) run and cleared the single GPU queue (I go after them).
- [ ] repo + env deployed to the box.

I will **flag before**: downloading the h5ad, deploying to the box, or submitting any GPU job. When those
prerequisites are met I'll submit strictly via `python gpu_queue.py submit jepa`.

## CP2 definition-of-done — status
- [x] JEPA to the exact §7e recipe, on Dev1's real encoder (learns; no collapse).
- [x] JEPA → real causal-encoder weight transfer (tested).
- [x] 2×2 harness, VOI + subsampling, all figures — built + tested on synthetic.
- [x] Queue entry points wired; 5 adversarial-review findings fixed.
- [ ] Runs against **real** data (blocked on G1 + cell cache + G2/G3 + box deploy — the GPU handshake above).
