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
            self.shard_paths = [self.cells_dir / s for s in self.manifest["shards"]]
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
# Task 2: materialize the 1–2M-cell cache using Developer 1's stratified sampler.
# ---------------------------------------------------------------------------
def materialize_cell_cache(config, h5ad_path=None, shard_size: int = 100_000) -> Path:
    """Draw ~``config.n_cells`` cells from the 22M, stratified over donor x condition,
    normalize to log1p-CP10k over the frozen HVG panel, and write the ``CELLS_DIR``
    cache (§7e / §2). Uses Developer 1's ``data.read_backed`` + ``stratified_cell_indices``.

    Requires the raw ``.h5ad`` (a large download); flagged, not run implicitly.
    """
    from core import contract, data as d1data, split as split_mod

    contract.ensure_dirs()
    h5ad_path = Path(h5ad_path) if h5ad_path else (contract.RAW_DIR / "GSE278572.h5ad")
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"raw h5ad not found at {h5ad_path}. The single-cell subsample needs the "
            "downloaded GSE278572 object (a >5 GB download — flag before fetching)."
        )
    adata = d1data.read_backed(h5ad_path)
    idx = d1data.stratified_cell_indices(
        adata.obs, config.n_cells, seed=config.seed, strata=("condition", "donor")
    )
    hvg = split_mod.load_hvg()
    # normalize + restrict to HVG columns in chunks (never densify the full matrix)
    mat = d1data.cells_to_hvg_matrix(adata, idx, hvg) if hasattr(d1data, "cells_to_hvg_matrix") else \
        _fallback_hvg_matrix(adata, idx, hvg)
    return write_cell_cache(contract.CELLS_DIR, mat, shard_size=shard_size)


def _fallback_hvg_matrix(adata, idx, hvg):
    """Minimal densify-in-chunks if data.py doesn't expose a helper. log1p-CP10k over HVG."""
    import numpy as np

    var_names = list(adata.var_names)
    col = {g: j for j, g in enumerate(var_names)}
    hvg_cols = [col[g] for g in hvg if g in col]
    out = np.zeros((len(idx), len(hvg)), dtype=np.float32)
    idx = np.sort(np.asarray(idx))
    for i0 in range(0, len(idx), 20_000):
        sl = idx[i0:i0 + 20_000]
        block = adata[sl].to_memory().X
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        libsize = block.sum(axis=1, keepdims=True)
        libsize[libsize == 0] = 1.0
        block = np.log1p(block / libsize * 1e4)
        out[i0:i0 + len(sl)] = block[:, hvg_cols].astype(np.float32)
    return out


__all__ = [
    "SyntheticHVGCells", "synthetic_priors", "CellCache", "write_cell_cache",
    "WindowedJEPALoader", "build_cell_loader", "materialize_cell_cache",
    "MANIFEST_NAME", "SHARD_GLOB",
]
