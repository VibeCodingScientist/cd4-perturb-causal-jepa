# Developer 2 — JEPA + analysis half (status + integration contract)

Built on branch `dev2` (worktree `../cd4-ws2`) against the frozen `core/contract.py`.
Everything is unit-tested on synthetic fixtures and swaps to real data / Developer 1's
real classes at **core-frozen** with no change to the JEPA training logic.

## What's here

| File | Role | Plan |
|---|---|---|
| `core/models/jepa.py` | Cell-JEPA pretraining: masked-value student, stop-grad EMA teacher, predictor, cosine + recon loss, VICReg collapse guard, measure-then-extrapolate gate, checkpointing, `python -m core.models.jepa` (G4) | §7e |
| `core/models/encoder_api.py` | The gene-token encoder interface both halves share (`EncoderOutput`, `GeneEncoder` protocol, mask sentinel) | §7d/§7e |
| `core/models/_reference_gene_tokens.py` | Reference `GeneTokenEncoder` (faithful CisTransCell-style, mask-aware) — stand-in until Dev 1's `gene_tokens.py`; doubles as the interface spec | §7d |
| `core/models/jepa_data.py` | Single-cell feed: `CellCacheDataset` (mmap shards), `SyntheticCellDataset`, `pad_collate`, `build_cell_loader` | §7e/§2 |
| `core/models/jepa_integration.py` | JEPA → causal weight transfer + the G5 seams (`jepa_causal`, `jepa_only`) | §7f |
| `core/voi.py` | Ensemble-disagreement VOI + subsampling / sample-efficiency curve | §7h |
| `core/ablation.py` | The 2×2 harness: score the four models, upsert the benchmark table, C2/C3 contrasts | §7f |
| `figures/make_figures.py` | Figures 1–4 (benchmark table, 2×2, sample efficiency, biology annotation) | §12 |
| `tests/` | 46 tests, all green: JEPA numerics, weight transfer, VOI, ablation, figures | — |

Run the suite: `.venv/bin/python -m pytest tests/ -q` (venv inherits system python3.12
packages + torch; see `requirements-dev.txt`).

## Interfaces Developer 1's code must satisfy (so integration is a drop-in)

### 1. `core/models/gene_tokens.py::GeneTokenEncoder`
`core.models.jepa` imports the real class if present and falls back to the reference
otherwise. For **weight transfer** (`student.state_dict()` → `CausalCisTransFormer.encoder`)
and for the JEPA student to be usable, the real encoder should expose (see
`core/models/encoder_api.py` and the reference for the exact shape):

- `__init__(n_genes, d_model, n_heads, n_layers, n_proxy, ...)`
- attribute `d_model: int`
- attribute `value_head: nn.Module` mapping token embeddings `[.., d_model] → [.., 1]`
  (JEPA reconstruction term)
- `forward(gene_ids [B,L] long, values [B,L] float, key_padding_mask=None, attn_mask=None)
  → EncoderOutput(tokens [B,L,d_model], pooled [B,d_model])`
- **`attn_mask` is where the causal do-mask (§7d) is applied**, on the *same* gene
  self-attention weights JEPA pretrains. JEPA passes `attn_mask=None`.

If Dev 1's signature differs, only the thin adapter in `jepa_integration` changes — the
JEPA loop is untouched. If submodule names differ, `load_jepa_into_encoder(..., strict=False)`
reports the mismatch instead of silently misloading.

### 2. `core/data.py` single-cell subsampler → `CELLS_DIR`
JEPA consumes `DATA_ROOT/cells/` in this schema (mirrored by
`jepa_data.write_synthetic_cell_cache`, and read by `CellCacheDataset`):

```
cells_*.npz  each: gene_ids int32 [N,L]  (Ensembl-id codes 0..n_genes-1, right-padded)
                   values   float32 [N,L] (log1p-CP10k; pad = 0.0)
                   lengths  int32   [N]   (real gene count per cell, ≤ 600)
manifest.json  {"n_cells","n_genes","max_genes","shards":[...]}
```
~1–2M cells, stratified over donor × condition (§7e/§2). Padding uses gene_id 0 (valid
embedding index) + a per-batch `key_padding_mask`.

### 3. `core/eval.py::evaluate(pred_delta_df, split) -> dict`
Consumed as-is by `core.ablation`. The harness reads `runs/<model>_<split>.parquet` and
calls `evaluate`; it never reimplements metrics. Tests inject a mock evaluate.

### 4. `core/models/causal_cistransformer.py` (G5 only)
`jepa_integration.build_jepa_causal_model` / `finetune_and_predict` expect
`CausalCisTransFormer(use_causal_mask: bool, ...)` with an `.encoder` attribute and a
trainer `train_causal_model(model, split, model_name, ...) -> pred_delta_df`. Until then
those functions raise a clear, actionable `ImportError` (they are the G5 seam).

## Integration steps at `core-frozen`
1. `git merge origin/main` into `dev2`.
2. Real `GeneTokenEncoder` is picked up automatically by `_default_encoder_cls`.
3. Dev 1 runs the subsampler → `CELLS_DIR`; `build_cell_loader` then feeds real cells.
4. Submit **G4** after Dev 1's G2/G3 clear the queue: `python gpu_queue.py submit jepa`
   → runs `core.models.jepa.main` (overnight; epoch-1 gate first; checkpoints
   `CHECKPOINTS_DIR/jepa_final.pt`).
5. Submit **G5**: fine-tune JEPA-init causal (`jepa_causal`, mask on) + `jepa_only`
   (mask off) via `jepa_integration.finetune_and_predict`, writing the two `runs/`.
6. `core.ablation.run_2x2()` scores all four cells, upserts `results/benchmark_table.csv`
   (won't clobber CP1 rows), returns the 2×2 + C2/C3 contrasts.
7. `core.voi` + `figures.make_figures.make_all_figures` → the three demo figures.

## CP2 definition-of-done — status
- [x] JEPA implemented to the exact §7e recipe (verified it learns + does not collapse).
- [x] JEPA → causal weight transfer implemented and tested.
- [x] 2×2 harness (`jepa_only`, `jepa_causal` rows) built against the eval signature.
- [x] VOI scores + subsampling curve built.
- [x] All three demo figures + biology figure render.
- [ ] Runs against **real** models/data (blocked on core-frozen: real encoder, cells,
      Dev 1's causal runs in `runs/`, and `snakemake` wiring — Dev 1 owns the Snakefile).
