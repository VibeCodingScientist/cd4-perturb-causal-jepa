"""Tests for the single-cell data feed (schema round-trip + collate + loader)."""
import numpy as np
import torch

from core.models.jepa_data import (
    CellCacheDataset,
    SyntheticCellDataset,
    build_cell_loader,
    pad_collate,
    write_synthetic_cell_cache,
)


def test_pad_collate_shapes_and_mask():
    batch = [
        (np.array([1, 2, 3], dtype=np.int32), np.array([0.1, 0.2, 0.3], dtype=np.float32), 3),
        (np.array([4, 5, 0], dtype=np.int32), np.array([0.4, 0.5, 0.0], dtype=np.float32), 2),
    ]
    gene_ids, values, kpm = pad_collate(batch)
    assert gene_ids.shape == (2, 3) and values.shape == (2, 3) and kpm.shape == (2, 3)
    assert gene_ids.dtype == torch.long and values.dtype == torch.float32 and kpm.dtype == torch.bool
    # row 0 fully real, row 1 has one pad at the end
    assert kpm[0].tolist() == [False, False, False]
    assert kpm[1].tolist() == [False, False, True]
    assert gene_ids[1, 2].item() == 0  # padded gene id
    assert torch.allclose(values[0], torch.tensor([0.1, 0.2, 0.3]))


def test_synthetic_dataset_deterministic_and_shaped():
    a = SyntheticCellDataset(n_cells=32, n_genes=48, max_genes=10, seed=0)
    b = SyntheticCellDataset(n_cells=32, n_genes=48, max_genes=10, seed=0)
    g0, v0, n0 = a[3]
    g1, v1, n1 = b[3]
    assert g0.shape == (10,) and v0.shape == (10,) and int(n0) == 10
    assert np.array_equal(g0, g1) and np.allclose(v0, v1)
    # gene ids within range and unique within a cell
    assert g0.max() < 48 and len(set(g0.tolist())) == 10
    assert (v0 >= 0).all()  # log1p-CP10k-like non-negative


def test_cell_cache_roundtrip_across_shards(tmp_path):
    cells_dir = tmp_path / "cells"
    write_synthetic_cell_cache(cells_dir, n_cells=100, n_genes=48, max_genes=8, shard_size=30, seed=2)
    ds = CellCacheDataset(cells_dir)
    assert len(ds) == 100
    # sample from a later shard (index past the first shard boundary)
    g, v, n = ds[95]
    assert g.shape == (8,) and v.shape == (8,) and int(n) == 8
    # every index is retrievable
    for i in (0, 29, 30, 59, 60, 99):
        gi, vi, ni = ds[i]
        assert gi.shape == (8,)


def test_build_cell_loader_yields_batches():
    class _Cfg:
        batch_size = 8

    ds = SyntheticCellDataset(n_cells=64, n_genes=48, max_genes=10, seed=0)
    loader = build_cell_loader(_Cfg(), dataset=ds)
    gene_ids, values, kpm = next(iter(loader))
    assert gene_ids.shape[0] == 8
    assert gene_ids.dtype == torch.long and values.dtype == torch.float32
