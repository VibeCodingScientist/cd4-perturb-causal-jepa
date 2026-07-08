"""Tests for the windowed single-cell feed + the mmap-able .npy cell cache."""
import numpy as np
import torch

from core.models.jepa_data import (
    CellCache,
    SyntheticHVGCells,
    WindowedJEPALoader,
    build_cell_loader,
    synthetic_priors,
    write_cell_cache,
)


class _Cfg:
    def __init__(self, batch_size=8, window=10, esm2_dim=16, ctx_dim=8, seed=0):
        self.batch_size = batch_size
        self.window = window
        self.esm2_dim = esm2_dim
        self.ctx_dim = ctx_dim
        self.seed = seed


def test_windowed_loader_shapes():
    cells = SyntheticHVGCells(n_cells=64, hvg_n=48, seed=0).matrix()
    esm2, ctx = synthetic_priors(48, 16, 8, seed=0)
    loader = WindowedJEPALoader(cells, esm2, ctx, batch_size=8, window=10, seed=0)
    values, e, c = next(iter(loader))
    assert values.shape == (8, 10) and e.shape == (10, 16) and c.shape == (10, 8)
    assert values.dtype == torch.float32 and e.dtype == torch.float32


def test_windowed_loader_samples_different_windows():
    cells = SyntheticHVGCells(n_cells=64, hvg_n=48, seed=0).matrix()
    esm2, ctx = synthetic_priors(48, 16, 8, seed=0)
    loader = WindowedJEPALoader(cells, esm2, ctx, batch_size=8, window=10, seed=0)
    it = iter(loader)
    _, e0, _ = next(it)
    _, e1, _ = next(it)
    # different gene windows -> different prior rows (with overwhelming probability)
    assert not torch.allclose(e0, e1)


def test_cell_cache_is_truly_memmapped(tmp_path):
    """Regression: the cache must memory-map (np.memmap), not load fully into RAM."""
    mat = SyntheticHVGCells(n_cells=250, hvg_n=48, seed=1).matrix()
    write_cell_cache(tmp_path, mat, shard_size=100)
    cache = CellCache(tmp_path)
    assert len(cache) == 250
    assert isinstance(cache._shard(0), np.memmap), "shards must be real memmaps"


def test_cell_cache_take_across_shards(tmp_path):
    mat = SyntheticHVGCells(n_cells=250, hvg_n=48, seed=2).matrix()
    write_cell_cache(tmp_path, mat, shard_size=100)
    cache = CellCache(tmp_path)
    # rows spanning shard boundaries (99|100, 199|200) must match the source matrix
    idx = np.array([0, 99, 100, 199, 200, 249])
    got = cache.take(idx)
    assert np.allclose(got, mat[idx])


def test_build_cell_loader_with_injected_arrays():
    cells = SyntheticHVGCells(n_cells=64, hvg_n=48, seed=0).matrix()
    esm2, ctx = synthetic_priors(48, 16, 8, seed=0)
    loader = build_cell_loader(_Cfg(), cells=cells, esm2=esm2, ctx=ctx)
    values, e, c = next(iter(loader))
    assert values.shape[0] == 8 and e.shape == (10, 16)


def test_loader_reads_from_disk_cache(tmp_path):
    mat = SyntheticHVGCells(n_cells=80, hvg_n=48, seed=3).matrix()
    write_cell_cache(tmp_path, mat, shard_size=40)
    cache = CellCache(tmp_path)
    esm2, ctx = synthetic_priors(48, 16, 8, seed=3)
    loader = build_cell_loader(_Cfg(), cells=cache, esm2=esm2, ctx=ctx)
    values, _, _ = next(iter(loader))
    assert values.shape == (8, 10)
