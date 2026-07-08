"""core.models.jepa_data — single-cell data feed for JEPA pretraining (§7e / §2).

Developer 1's ``core/data.py`` subsampler draws ~1-2M cells from the 22M, stratified
over donor x condition, and writes them to ``CELLS_DIR`` (``DATA_ROOT/cells/``).
Developer 2 only *consumes* that cache. To let both sides agree, the on-disk schema
is pinned here and mirrored by ``write_synthetic_cell_cache`` (used in tests and as a
runnable dev fixture).

On-disk cell cache (``CELLS_DIR``)
----------------------------------
One or more compressed shards ``cells_*.npz``, each holding a block of cells:

    gene_ids : int32   [N, L]   Ensembl-id integer codes, 0..n_genes-1, right-padded
    values   : float32 [N, L]   log1p-CP10k expression (pad positions = 0.0)
    lengths  : int32   [N]      number of real (non-pad) genes per cell (<= L <= 600)
    donor    : int8    [N]       (optional) donor index, for provenance
    condition: int8    [N]       (optional) condition index, for provenance

plus ``manifest.json``: {"n_cells", "n_genes", "max_genes", "shards": [...]}.

Padding uses gene_id 0 (a valid embedding index) and value 0.0; a per-batch
``key_padding_mask`` (True == pad) keeps padded positions out of attention, pooling,
and the reconstruction target. Shards are memory-mapped, never fully materialized —
important on the 24 GB L4 (and essential on an 8 GB laptop during dev).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MANIFEST_NAME = "manifest.json"
SHARD_GLOB = "cells_*.npz"


# ---------------------------------------------------------------------------
# Collate: pad a list of (gene_ids, values, length) to the batch's max length.
# ---------------------------------------------------------------------------
def pad_collate(batch):
    """-> (gene_ids [B, Lmax] long, values [B, Lmax] float, key_padding_mask [B, Lmax] bool)."""
    lengths = [int(item[2]) for item in batch]
    lmax = max(lengths)
    b = len(batch)
    gene_ids = torch.zeros(b, lmax, dtype=torch.long)
    values = torch.zeros(b, lmax, dtype=torch.float32)
    kpm = torch.ones(b, lmax, dtype=torch.bool)          # True == pad
    for i, (g, v, n) in enumerate(batch):
        n = int(n)
        gene_ids[i, :n] = torch.as_tensor(g[:n], dtype=torch.long)
        values[i, :n] = torch.as_tensor(v[:n], dtype=torch.float32)
        kpm[i, :n] = False
    return gene_ids, values, kpm


# ---------------------------------------------------------------------------
# Synthetic dataset (tests + dev): correlated gene expression so JEPA has real
# structure to learn (masking one gene is predictable from its correlates).
# ---------------------------------------------------------------------------
class SyntheticCellDataset(Dataset):
    """In-memory synthetic cells with low-rank structure (learnable, non-trivial)."""

    def __init__(
        self,
        n_cells: int = 2048,
        n_genes: int = 512,
        max_genes: int = 64,
        rank: int = 8,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.n_genes = n_genes
        self.max_genes = max_genes
        # low-rank latent -> gene loadings gives correlated expression
        loadings = rng.standard_normal((n_genes, rank)).astype(np.float32)
        latents = rng.standard_normal((n_cells, rank)).astype(np.float32)
        base = latents @ loadings.T                       # [n_cells, n_genes]
        self._gene_ids = np.empty((n_cells, max_genes), dtype=np.int32)
        self._values = np.empty((n_cells, max_genes), dtype=np.float32)
        self._lengths = np.full(n_cells, max_genes, dtype=np.int32)
        for c in range(n_cells):
            genes = rng.choice(n_genes, size=max_genes, replace=False)
            self._gene_ids[c] = genes
            vals = base[c, genes] + 0.1 * rng.standard_normal(max_genes).astype(np.float32)
            # log1p-CP10k-like: non-negative
            self._values[c] = np.log1p(np.abs(vals))

    def __len__(self):
        return len(self._gene_ids)

    def __getitem__(self, i):
        return self._gene_ids[i], self._values[i], self._lengths[i]


# ---------------------------------------------------------------------------
# On-disk cache dataset (real run): memory-mapped shards.
# ---------------------------------------------------------------------------
class CellCacheDataset(Dataset):
    """Reads the ``CELLS_DIR`` shard schema above with memory-mapped numpy."""

    def __init__(self, cells_dir: Path):
        self.cells_dir = Path(cells_dir)
        manifest_path = self.cells_dir / MANIFEST_NAME
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            shard_names = self.manifest["shards"]
            self.shard_paths = [self.cells_dir / s for s in shard_names]
        else:
            self.shard_paths = sorted(self.cells_dir.glob(SHARD_GLOB))
            if not self.shard_paths:
                raise FileNotFoundError(
                    f"no cell cache in {self.cells_dir} (expected {MANIFEST_NAME} or {SHARD_GLOB}). "
                    "Run Developer 1's data.py subsampler first, or use SyntheticCellDataset."
                )
            self.manifest = None
        # index: cumulative cell counts across shards (mmap, don't load values)
        self._shard_lengths = []
        self._mmaps: list[Optional[dict]] = [None] * len(self.shard_paths)
        for sp in self.shard_paths:
            with np.load(sp, mmap_mode="r") as z:
                self._shard_lengths.append(z["gene_ids"].shape[0])
        self._cumsum = np.cumsum([0] + self._shard_lengths)
        self.n_genes = int(self.manifest["n_genes"]) if self.manifest else None

    def _shard(self, si: int) -> dict:
        if self._mmaps[si] is None:
            z = np.load(self.shard_paths[si], mmap_mode="r")
            self._mmaps[si] = {"gene_ids": z["gene_ids"], "values": z["values"], "lengths": z["lengths"]}
        return self._mmaps[si]

    def __len__(self):
        return int(self._cumsum[-1])

    def __getitem__(self, i):
        si = int(np.searchsorted(self._cumsum, i, side="right") - 1)
        local = i - int(self._cumsum[si])
        z = self._shard(si)
        return z["gene_ids"][local], z["values"][local], z["lengths"][local]


def write_synthetic_cell_cache(
    cells_dir: Path,
    n_cells: int = 4096,
    n_genes: int = 512,
    max_genes: int = 64,
    shard_size: int = 1024,
    seed: int = 0,
) -> Path:
    """Materialize a synthetic cache in the real on-disk schema (dev fixture)."""
    cells_dir = Path(cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    ds = SyntheticCellDataset(n_cells, n_genes, max_genes, seed=seed)
    shards = []
    for start in range(0, n_cells, shard_size):
        end = min(start + shard_size, n_cells)
        name = f"cells_{start:08d}.npz"
        np.savez_compressed(
            cells_dir / name,
            gene_ids=ds._gene_ids[start:end],
            values=ds._values[start:end],
            lengths=ds._lengths[start:end],
        )
        shards.append(name)
    manifest = {"n_cells": n_cells, "n_genes": n_genes, "max_genes": max_genes, "shards": shards}
    (cells_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return cells_dir


def build_cell_loader(config, dataset: Optional[Dataset] = None) -> DataLoader:
    """DataLoader of ``(gene_ids, values, key_padding_mask)`` batches for pretraining.

    Prefers the real ``CELLS_DIR`` cache; ``dataset`` overrides (tests inject
    ``SyntheticCellDataset``). Never loads the whole cache into RAM.
    """
    if dataset is None:
        from core import contract
        dataset = CellCacheDataset(contract.CELLS_DIR)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        drop_last=True,
        num_workers=0,
    )


__all__ = [
    "pad_collate", "SyntheticCellDataset", "CellCacheDataset",
    "write_synthetic_cell_cache", "build_cell_loader",
    "MANIFEST_NAME", "SHARD_GLOB",
]
