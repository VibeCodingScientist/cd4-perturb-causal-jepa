"""Synthetic fixtures shared across the Developer-2 tests.

Everything here is small + deterministic. No real data, no trained models: JEPA runs
on tiny reference encoders, the ablation/VOI/figure layers run on synthetic run files
and a mock ``evaluate``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from core import contract


# ---------------------------------------------------------------------------
# Tiny reference encoder + JEPA for the numeric tests
# ---------------------------------------------------------------------------
def tiny_encoder(n_genes: int = 64, d_model: int = 32, seed: int = 0):
    from core.models._reference_gene_tokens import ReferenceGeneTokenEncoder

    torch.manual_seed(seed)
    return ReferenceGeneTokenEncoder(
        n_genes=n_genes, d_model=d_model, n_heads=2, n_layers=2, n_proxy=4
    )


def tiny_jepa(n_genes: int = 64, d_model: int = 32, seed: int = 0, **kwargs):
    from core.models.jepa import CellJEPA

    enc = tiny_encoder(n_genes, d_model, seed)
    return CellJEPA(enc, d_model, **kwargs)


def random_cell_batch(batch: int = 8, length: int = 16, n_genes: int = 64, seed: int = 0, pad: bool = False):
    """(gene_ids [B,L] long, values [B,L] float, key_padding_mask [B,L] bool | None)."""
    g = torch.Generator().manual_seed(seed)
    gene_ids = torch.randint(0, n_genes, (batch, length), generator=g)
    values = torch.rand(batch, length, generator=g)
    kpm = None
    if pad:
        kpm = torch.zeros(batch, length, dtype=torch.bool)
        kpm[:, length // 2:] = True  # second half padded
    return gene_ids, values, kpm


# ---------------------------------------------------------------------------
# Synthetic prediction (delta) frames + run files
# ---------------------------------------------------------------------------
def make_delta_df(n_perts: int = 12, n_genes: int = 20, constant: float | None = None, seed: int = 0) -> pd.DataFrame:
    """A predicted-delta frame: index = pert_id, columns = gene ids, values = delta.

    ``constant`` fills every cell with the same value (so the mock evaluate maps it to
    a known score — used to control 2x2 ordering in tests).
    """
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
    """Deterministic stand-in for ``core.eval.evaluate`` (frozen signature).

    Maps the mean absolute predicted delta monotonically to each metric, so a
    mode-collapsed (near-zero) prediction scores low Pearson / high perturbench_rank
    (flagged), and larger, more confident deltas score better. Enough structure to
    unit-test the ablation + figure layers without trained models.
    """
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
