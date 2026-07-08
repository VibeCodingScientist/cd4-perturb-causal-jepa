"""Tests for single-cell ingestion from the assigned_guide schema (Task 2)."""
import json

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

from core.models.jepa_data import (
    CellCache,
    append_cells_to_cache,
    ingest_assigned_guide,
)


def _synthetic_assigned_guide(n_cells=500, n_genes=40, n_low_quality=50, seed=0):
    """Mimic a D*_*.assigned_guide.h5ad: raw-count CSR X, obs.low_quality,
    var.gene_ids (Ensembl) with a symbol var index."""
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32))
    var = pd.DataFrame(
        {"gene_ids": [f"ENSG{900000 + j:06d}" for j in range(n_genes)], "gene_name": [f"SYM{j}" for j in range(n_genes)]},
        index=[f"SYM{j}" for j in range(n_genes)],
    )
    lq = np.array([False] * (n_cells - n_low_quality) + [True] * n_low_quality)
    # perturbed_gene_id cycles over a small set so holdout filtering is testable
    pgid = np.array([f"ENSG{900000 + (i % n_genes):06d}" for i in range(n_cells)])
    obs = pd.DataFrame(
        {"low_quality": lq, "guide_id": ["g0"] * n_cells, "perturbed_gene_id": pgid},
        index=[f"c{i}" for i in range(n_cells)],
    )
    return anndata.AnnData(X=X, obs=obs, var=var)


def test_ingest_shape_and_hvg_reindex():
    adata = _synthetic_assigned_guide()
    # 5 present HVG + 1 absent -> absent column is all zeros
    hvg = [f"ENSG{900000 + j:06d}" for j in (0, 1, 2, 5, 10)] + ["ENSG000INVALID"]
    mat = ingest_assigned_guide(adata, hvg, n_cells=100, seed=1)
    assert mat.shape == (100, 6)
    assert np.isfinite(mat).all() and (mat >= 0).all()
    assert (mat[:, 5] == 0).all(), "absent HVG must map to a zero column"


def test_ingest_filters_low_quality():
    adata = _synthetic_assigned_guide(n_cells=500, n_low_quality=50)
    hvg = [f"ENSG{900000 + j:06d}" for j in range(5)]
    # asking for more than the 450 good cells returns exactly the good cells (LQ dropped)
    mat = ingest_assigned_guide(adata, hvg, n_cells=10_000)
    assert mat.shape[0] == 450


def test_ingest_drops_gene_holdout_cells():
    # cells cycle perturbed_gene_id over ENSG900000..; hold out two of them
    adata = _synthetic_assigned_guide(n_cells=600, n_genes=40, n_low_quality=0)
    hvg = [f"ENSG{900000 + j:06d}" for j in range(5)]
    holdout = ["ENSG900003", "ENSG900007"]  # version-suffix-free, matches _strip_version
    mat = ingest_assigned_guide(adata, hvg, n_cells=10_000, holdout_genes=holdout)
    # 600 cells, 40 genes cycled -> 15 cells per gene; 2 held-out genes -> 30 cells dropped
    assert mat.shape[0] == 600 - 30


def test_ingest_normalization_is_log1p_cp10k():
    # one good cell, counts [10, 0, 90] over 3 genes -> CP10k [1000,0,9000] -> log1p
    X = sparse.csr_matrix(np.array([[10.0, 0.0, 90.0]], dtype=np.float32))
    var = pd.DataFrame({"gene_ids": ["ENSGA", "ENSGB", "ENSGC"]}, index=["A", "B", "C"])
    obs = pd.DataFrame({"low_quality": [False]}, index=["c0"])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    mat = ingest_assigned_guide(adata, ["ENSGA", "ENSGB", "ENSGC"], n_cells=1)
    expected = np.log1p(np.array([1000.0, 0.0, 9000.0], dtype=np.float32))
    assert np.allclose(mat[0], expected, atol=1e-4)


def test_append_records_provenance_and_reads_back(tmp_path):
    adata = _synthetic_assigned_guide(seed=1)
    hvg = [f"ENSG{900000 + j:06d}" for j in range(8)]
    m1 = ingest_assigned_guide(adata, hvg, n_cells=120, seed=1)
    m2 = ingest_assigned_guide(_synthetic_assigned_guide(seed=2), hvg, n_cells=80, seed=2)
    append_cells_to_cache(tmp_path, m1, donor="D1", condition="Rest")
    append_cells_to_cache(tmp_path, m2, donor="D2", condition="Stim8hr")

    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["n_cells"] == 200 and man["hvg_n"] == 8
    assert man["shards"][0]["donor"] == "D1" and man["shards"][0]["condition"] == "Rest"
    assert man["shards"][1]["donor"] == "D2" and man["shards"][1]["condition"] == "Stim8hr"

    cache = CellCache(tmp_path)
    assert len(cache) == 200
    # take one row from each shard (crosses the shard boundary at index 120)
    got = cache.take(np.array([0, 120, 199]))
    assert got.shape == (3, 8) and np.allclose(got[0], m1[0]) and np.allclose(got[1], m2[0])
