"""Synthetic fixtures shared across the Developer-2 tests.

Everything is small + deterministic and now runs on Developer 1's **real**
``GeneTokenEncoder`` (post core-frozen) with tiny prior dims, so the tests exercise the
actual integration path — not a stand-in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from core import contract


# ---------------------------------------------------------------------------
# Tiny real encoder + JEPA for the numeric tests
# ---------------------------------------------------------------------------
def tiny_encoder(d_model: int = 32, esm2_dim: int = 16, ctx_dim: int = 8, n_proxy: int = 4, seed: int = 0):
    from core.models.gene_tokens import GeneTokenEncoder

    torch.manual_seed(seed)
    return GeneTokenEncoder(d_model=d_model, esm2_dim=esm2_dim, ctx_dim=ctx_dim, n_proxy=n_proxy, n_heads=2)


def tiny_jepa(d_model: int = 32, esm2_dim: int = 16, ctx_dim: int = 8, seed: int = 0, **kw):
    from core.models.jepa import CellJEPA

    enc = tiny_encoder(d_model, esm2_dim, ctx_dim, seed=seed)
    return CellJEPA(enc, d_model, **kw)


def random_window_batch(batch: int = 8, window: int = 16, esm2_dim: int = 16, ctx_dim: int = 8, seed: int = 0):
    """(values [B,W], esm2 [W,esm2_dim], ctx [W,ctx_dim]) — the encoder's input shape."""
    g = torch.Generator().manual_seed(seed)
    values = torch.rand(batch, window, generator=g)
    esm2 = torch.rand(window, esm2_dim, generator=g)
    ctx = torch.rand(window, ctx_dim, generator=g)
    return values, esm2, ctx


# ---------------------------------------------------------------------------
# Synthetic prediction (delta) frames + run files
# ---------------------------------------------------------------------------
def make_delta_df(n_perts: int = 12, n_genes: int = 20, constant: float | None = None, seed: int = 0) -> pd.DataFrame:
    """A predicted-delta frame: index = pert_id, columns = gene ids, values = delta."""
    perts = [f"ENSG{100000 + i:06d}" for i in range(n_perts)]
    genes = [f"ENSG{200000 + j:06d}" for j in range(n_genes)]
    if constant is not None:
        data = np.full((n_perts, n_genes), float(constant))
    else:
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_perts, n_genes))
    df = pd.DataFrame(data, index=perts, columns=genes)
    df.index.name = "pert_id"
    return df


def write_run(model_name: str, split: str, df: pd.DataFrame) -> str:
    contract.ensure_dirs()
    path = contract.run_path(model_name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return str(path)


def mock_evaluate(pred_df: pd.DataFrame, split: str) -> dict:
    """Deterministic stand-in for ``core.eval.evaluate`` (frozen signature). Maps the
    mean absolute predicted delta monotonically to each metric."""
    v = np.abs(pred_df.to_numpy(dtype=float))
    mag = float(v.mean()) if v.size else 0.0
    pear = float(np.tanh(mag))
    return {
        contract.METRIC_PEARSON_DELTA: pear,
        contract.METRIC_PERTURBENCH_RANK: float(max(0.0, 0.5 - 0.3 * mag)),
        contract.METRIC_DES: float(min(1.0, 0.5 + 0.4 * mag)),
        contract.METRIC_MAE: float(0.5 / (1.0 + mag)),
        contract.METRIC_SPEARMAN_LFC: pear * 0.9,
        contract.METRIC_SPEARMAN_EFFECT: pear * 0.8,
        contract.METRIC_AUPRC: float(min(1.0, 0.4 + 0.3 * mag)),
        contract.METRIC_EDISTANCE: float(1.0 / (1.0 + mag)),
    }
