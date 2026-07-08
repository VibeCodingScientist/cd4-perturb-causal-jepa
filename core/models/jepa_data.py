"""core.models.jepa_data — single-cell feed for JEPA pretraining (§7e / §2).

Developer 1's ``core/data.py`` provides the *index* sampler
(``stratified_cell_indices``) + backed reader (``read_backed``); **materializing** the
1–2M-cell JEPA cache into ``CELLS_DIR`` is Developer 2's Task 2, done here by
``materialize_cell_cache``.

Feed shape (matches ``GeneTokenEncoder.forward(values, esm2, ctx)``)
--------------------------------------------------------------------
Each JEPA batch is a **gene window** (<=600 HVG, §7e) shared across the batch — the
same gene-window batching the causal model uses:

    values : (B, W)  each cell's log1p-CP10k expression over the window's genes
    esm2   : (W, esm2_dim)  window genes' ESM-2 prior     (batch-shared)
    ctx    : (W, ctx_dim)   window genes' context prior    (batch-shared)

The student masks a fraction of ``values`` within each cell; the teacher sees them
unmasked (§7e).

On-disk cache (``CELLS_DIR``) — mmap-friendly ``.npy`` (NOT ``.npz``)
--------------------------------------------------------------------
    cells_XXXX.npy   float32 [N, HVG]   cells' expression over the FULL HVG panel
    manifest.json    {"n_cells","hvg_n","shards":[...],"dtype":"float32"}

Single-array ``.npy`` shards are *genuinely* memory-mapped (``np.load(mmap_mode='r')``
returns an ``np.memmap``); a ``.npz`` archive cannot be mmapped by numpy, so a row
slice would fault the whole (decompressed) shard into RAM. Gene order is the frozen
HVG list, so ``esm2``/``ctx`` are gathered by column index — no per-cell gene ids.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

MANIFEST_NAME = "manifest.json"
SHARD_GLOB = "cells_*.npy"


# ---------------------------------------------------------------------------
# Synthetic cells (tests + dev): low-rank structure so masking is learnable.
# ---------------------------------------------------------------------------
class SyntheticHVGCells:
    """In-memory synthetic cells: expression over the full HVG panel, correlated via a
    low-rank latent so a masked gene is predictable from its correlates."""

    def __init__(self, n_cells: int = 2048, hvg_n: int = 256, rank: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hvg_n = hvg_n
        loadings = rng.standard_normal((hvg_n, rank)).astype(np.float32)
        latents = rng.standard_normal((n_cells, rank)).astype(np.float32)
        base = latents @ loadings.T + 0.1 * rng.standard_normal((n_cells, hvg_n)).astype(np.float32)
        self.expr = np.log1p(np.abs(base)).astype(np.float32)   # [n_cells, hvg_n], >= 0

    def __len__(self):
        return len(self.expr)

    def matrix(self) -> np.ndarray:
        return self.expr


def synthetic_priors(hvg_n: int, esm2_dim: int, ctx_dim: int, seed: int = 0):
    """Random per-gene ESM-2 / context priors [hvg_n, dim] for tests."""
    rng = np.random.default_rng(seed + 777)
    esm2 = rng.standard_normal((hvg_n, esm2_dim)).astype(np.float32)
    ctx = rng.standard_normal((hvg_n, ctx_dim)).astype(np.float32)
    return esm2, ctx


# ---------------------------------------------------------------------------
# On-disk cache (real run): mmap-able .npy shards.
# ---------------------------------------------------------------------------
class CellCache:
    """Memory-mapped access to the ``CELLS_DIR`` ``.npy`` shard schema.

    Only the touched pages of a shard are resident (real ``np.memmap``), so 1–2M cells
    never load into RAM at once — safe on the 8 GB dev laptop and lean on the L4.
    """

    def __init__(self, cells_dir: Path):
        self.cells_dir = Path(cells_dir)
        manifest_path = self.cells_dir / MANIFEST_NAME
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            # shard entries are either bare names (str) or provenance dicts {"name",...}
            names = [s["name"] if isinstance(s, dict) else s for s in self.manifest["shards"]]
            self.shard_paths = [self.cells_dir / n for n in names]
            self.hvg_n = int(self.manifest["hvg_n"])
        else:
            self.shard_paths = sorted(self.cells_dir.glob(SHARD_GLOB))
            if not self.shard_paths:
                raise FileNotFoundError(
                    f"no cell cache in {self.cells_dir} (expected {MANIFEST_NAME} or {SHARD_GLOB}). "
                    "Run materialize_cell_cache() first, or use SyntheticHVGCells."
                )
            self.manifest = None
            self.hvg_n = None
        # lazily-opened memmaps (one per shard; a memmap is cheap — no data read yet)
        self._shards: list[Optional[np.memmap]] = [None] * len(self.shard_paths)
        self._lengths = []
        for sp in self.shard_paths:
            mm = np.load(sp, mmap_mode="r")
            self._lengths.append(mm.shape[0])
            if self.hvg_n is None:
                self.hvg_n = int(mm.shape[1])
        self._cumsum = np.cumsum([0] + self._lengths)

    def _shard(self, si: int) -> np.memmap:
        if self._shards[si] is None:
            self._shards[si] = np.load(self.shard_paths[si], mmap_mode="r")
        return self._shards[si]

    def __len__(self):
        return int(self._cumsum[-1])

    def take(self, indices: np.ndarray) -> np.ndarray:
        """Gather a set of cells (rows) across shards into a dense [k, HVG] array."""
        out = np.empty((len(indices), self.hvg_n), dtype=np.float32)
        for i, gi in enumerate(indices):
            si = int(np.searchsorted(self._cumsum, gi, side="right") - 1)
            local = int(gi - self._cumsum[si])
            out[i] = self._shard(si)[local]
        return out


def write_cell_cache(cells_dir: Path, matrix: np.ndarray, shard_size: int = 100_000) -> Path:
    """Write a [n_cells, HVG] float32 matrix as mmap-able ``.npy`` shards + manifest."""
    cells_dir = Path(cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    n, hvg_n = matrix.shape
    shards = []
    for start in range(0, n, shard_size):
        end = min(start + shard_size, n)
        name = f"cells_{start:010d}.npy"
        np.save(cells_dir / name, np.ascontiguousarray(matrix[start:end], dtype=np.float32))
        shards.append(name)
    (cells_dir / MANIFEST_NAME).write_text(
        json.dumps({"n_cells": int(n), "hvg_n": int(hvg_n), "shards": shards, "dtype": "float32"}, indent=2)
    )
    return cells_dir


# ---------------------------------------------------------------------------
# Windowed batch loader: (values [B,W], esm2 [W,d], ctx [W,d]).
# ---------------------------------------------------------------------------
class WindowedJEPALoader:
    """Yields ``steps`` (or one epoch of) gene-window batches from a cell source.

    ``cells`` is a ``CellCache`` or a dense [n_cells, HVG] ndarray; ``esm2``/``ctx`` are
    [HVG, dim] prior matrices. Each batch samples ``batch_size`` cells and a random gene
    window of ``window`` columns, returning torch tensors ready for the encoder.
    """

    def __init__(self, cells, esm2: np.ndarray, ctx: np.ndarray, *, batch_size: int,
                 window: int, n_batches: Optional[int] = None, seed: int = 0):
        self.cells = cells
        self.esm2 = np.asarray(esm2, dtype=np.float32)
        self.ctx = np.asarray(ctx, dtype=np.float32)
        self.n_cells = len(cells)
        self.hvg_n = self.esm2.shape[0]
        self.batch_size = batch_size
        self.window = min(window, self.hvg_n)
        self.n_batches = n_batches if n_batches is not None else max(1, self.n_cells // batch_size)
        self.rng = np.random.default_rng(seed)

    def _cell_matrix(self, idx: np.ndarray) -> np.ndarray:
        if isinstance(self.cells, np.ndarray):
            return self.cells[idx]
        return self.cells.take(idx)                      # CellCache (mmap)

    def __iter__(self) -> Iterator:
        for _ in range(self.n_batches):
            cell_idx = self.rng.integers(0, self.n_cells, size=self.batch_size)
            win = self.rng.choice(self.hvg_n, size=self.window, replace=False)
            values = self._cell_matrix(cell_idx)[:, win]          # [B, W]
            esm2 = self.esm2[win]                                 # [W, esm2_dim]
            ctx = self.ctx[win]                                   # [W, ctx_dim]
            yield (
                torch.from_numpy(np.ascontiguousarray(values)),
                torch.from_numpy(np.ascontiguousarray(esm2)),
                torch.from_numpy(np.ascontiguousarray(ctx)),
            )


def build_cell_loader(config, cells=None, esm2=None, ctx=None) -> WindowedJEPALoader:
    """Windowed JEPA loader. Prefers the real ``CELLS_DIR`` cache + feature caches;
    ``cells``/``esm2``/``ctx`` override (tests inject synthetic)."""
    if cells is None:
        from core import contract
        cells = CellCache(contract.CELLS_DIR)
    if esm2 is None or ctx is None:
        from core import contract, features as feat
        hvg = _load_hvg_order()
        esm2 = feat.load_esm2().reindex(hvg).fillna(0.0).to_numpy(dtype=np.float32)
        ctx = feat.load_context_prior().reindex(hvg).fillna(0.0).to_numpy(dtype=np.float32)
    return WindowedJEPALoader(
        cells, esm2, ctx,
        batch_size=config.batch_size, window=config.window, seed=config.seed,
    )


def _load_hvg_order():
    """The frozen HVG gene order (columns of the cell cache)."""
    from core import contract
    return [g.strip() for g in contract.HVG_LIST_PATH.read_text().splitlines() if g.strip()]


# ---------------------------------------------------------------------------
# Task 2: ingest the per-(donor,condition) single-cell files into CELLS_DIR.
#
# The single cells live in `s3://.../D{donor}_{condition}.assigned_guide.h5ad`
# (public, no creds; 110-161 GB EACH). Donor/condition come from the FILENAME, not
# obs. Disk holds only ~one file at a time, so the box orchestrator
# (scripts/fetch_jepa_cells.py) does download -> ingest -> delete per file; this
# module owns the (tested) ingest + append.
#
# Per the readme, `.obs` has `low_quality` (filter), `guide_id` ("multi-guide" if
# ambiguous), `guide_type`; `.var` has `gene_ids` (Ensembl); `.X` is raw UMI counts
# (CSR). We filter low-quality cells, subsample, normalize log1p-CP10k, and reindex
# to the frozen HVG panel.
# ---------------------------------------------------------------------------
def _strip_version(ids) -> np.ndarray:
    """ENSG00000123456.5 -> ENSG00000123456 so single-cell var ids and the frozen HVG
    list join even if one carries version suffixes."""
    return np.array([str(g).split(".", 1)[0] for g in ids], dtype=object)


def _var_ensembl(adata) -> np.ndarray:
    """Per-var Ensembl id (version-stripped): prefer `.var['gene_ids']`, else var index."""
    ids = adata.var["gene_ids"] if "gene_ids" in adata.var.columns else adata.var_names
    return _strip_version(np.asarray(ids))


def _as_bool(series) -> np.ndarray:
    """Robust bool coercion for an obs column that may be bool / 0-1 / 'True'/'False'."""
    if series.dtype == bool:
        return series.to_numpy()
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "1.0", "yes"]).to_numpy()


def ingest_assigned_guide(
    adata,
    hvg: list,
    n_cells: int,
    seed: int = 42,
    filter_low_quality: bool = True,
    holdout_genes=None,
    chunk: int = 20_000,
    log_fn=None,
) -> np.ndarray:
    """Subsample + normalize cells from one `assigned_guide` AnnData to a [k, |HVG|]
    log1p-CP10k float32 matrix over the frozen HVG panel.

    ``holdout_genes``: Ensembl ids of the gene-hold-out set. Cells whose
    ``perturbed_gene_id`` is in this set are DROPPED — otherwise JEPA pretraining sees
    the held-out genes' knockdown phenotypes and the gene-hold-out C3 claim leaks
    (the condition/donor hold-outs are handled at the file level in fetch_jepa_cells).

    Works on a backed or in-memory AnnData; reads only the selected rows, in chunks,
    so the full (possibly 100+ GB) matrix is never densified.
    """
    obs = adata.obs
    keep = np.ones(adata.n_obs, dtype=bool)
    if filter_low_quality and "low_quality" in obs.columns:
        keep &= ~_as_bool(obs["low_quality"])
    if holdout_genes and "perturbed_gene_id" in obs.columns:
        hg = set(_strip_version(list(holdout_genes)).tolist())
        pg = _strip_version(obs["perturbed_gene_id"].astype(str).to_numpy())
        n_before = int(keep.sum())
        keep &= ~np.isin(pg, list(hg))
        if log_fn is not None:
            log_fn(f"[ingest] gene-holdout filter: dropped {n_before - int(keep.sum())} held-out-gene cells")
    pool = np.flatnonzero(keep)
    if len(pool) == 0:
        raise ValueError("no cells passed the low_quality filter")
    rng = np.random.default_rng(seed)
    take = min(n_cells, len(pool))
    sel = np.sort(rng.choice(pool, size=take, replace=False))

    ens = _var_ensembl(adata)
    col = {g: j for j, g in enumerate(ens)}
    hvg_stripped = _strip_version(hvg)
    hvg_cols = np.array([col.get(g, -1) for g in hvg_stripped])
    present = hvg_cols >= 0
    src_cols = hvg_cols[present]
    if log_fn is not None:
        log_fn(f"[ingest] HVG coverage: {int(present.sum())}/{len(hvg)} genes matched in var")

    out = np.zeros((len(sel), len(hvg)), dtype=np.float32)
    for i0 in range(0, len(sel), chunk):
        blk = sel[i0:i0 + chunk]
        sub = adata[blk]
        X = sub.to_memory().X if hasattr(sub, "to_memory") else sub.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X = X.astype(np.float32)
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        Xn = np.log1p(X / lib * 1e4)
        out[i0:i0 + len(blk)][:, present] = Xn[:, src_cols]
    return out


def append_cells_to_cache(cells_dir, matrix: np.ndarray, donor=None, condition=None) -> Path:
    """Append a [n, HVG] matrix as one more mmap-able shard, recording donor/condition
    provenance in the manifest. Multiple (donor,condition) files thus stratify the
    cache by composition (the loader then samples uniformly across cells)."""
    cells_dir = Path(cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cells_dir / MANIFEST_NAME
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        shards = [s if isinstance(s, dict) else {"name": s} for s in man.get("shards", [])]
        hvg_n = int(man["hvg_n"])
        if matrix.shape[1] != hvg_n:
            raise ValueError(f"matrix has {matrix.shape[1]} genes; cache HVG panel is {hvg_n}")
    else:
        man, shards, hvg_n = {}, [], int(matrix.shape[1])
    name = f"cells_{len(shards):04d}_{donor or 'NA'}_{condition or 'NA'}.npy"
    np.save(cells_dir / name, np.ascontiguousarray(matrix, dtype=np.float32))
    shards.append({"name": name, "n": int(matrix.shape[0]), "donor": donor, "condition": condition})
    man.update({
        "hvg_n": hvg_n, "dtype": "float32", "shards": shards,
        "n_cells": int(sum(s["n"] for s in shards)),
    })
    manifest_path.write_text(json.dumps(man, indent=2))
    return cells_dir


def ingest_file_to_cache(h5ad_path, hvg, n_cells, donor, condition, cells_dir=None, seed=42,
                         holdout_genes=None) -> int:
    """Read one assigned_guide h5ad (backed), subsample, and append to CELLS_DIR.
    Returns the number of cells added. The box orchestrator deletes the raw file after."""
    import anndata

    from core import contract

    cells_dir = cells_dir or contract.CELLS_DIR
    adata = anndata.read_h5ad(h5ad_path, backed="r")
    mat = ingest_assigned_guide(adata, hvg, n_cells, seed=seed, holdout_genes=holdout_genes, log_fn=print)
    append_cells_to_cache(cells_dir, mat, donor=donor, condition=condition)
    return int(mat.shape[0])


__all__ = [
    "SyntheticHVGCells", "synthetic_priors", "CellCache", "write_cell_cache",
    "WindowedJEPALoader", "build_cell_loader",
    "ingest_assigned_guide", "append_cells_to_cache", "ingest_file_to_cache",
    "MANIFEST_NAME", "SHARD_GLOB",
]
