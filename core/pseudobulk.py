"""
core.pseudobulk — (pert_id, condition, donor) mean profiles + matched-control deltas
(UNIFIED_BUILD_PLAN.md §7a).

Never materializes the full dense matrix (the real dataset is ~22M cells). Instead a
streaming accumulator sums expression per group over backed/chunked reads, then divides
by the count. delta = pert_pseudobulk - matched control (SAME condition, SAME donor).

Output: the frozen two-block pseudobulk parquets (`expr` + `delta`, see
contract.build_pseudobulk_frame), split into train/test by `core.split.is_train_row`.

The accumulator is fed `(obs_chunk, X_chunk)` pairs, so it works identically whether the
chunks come from a backed AnnData (production, `iter_anndata_chunks`) or from an
in-memory synthetic frame (tests). anndata/scanpy are imported lazily and only on the
production path.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import contract as C
from . import split as split_mod

GroupKey = Tuple[str, str, str]  # (pert_id, condition, donor)


class PseudobulkAccumulator:
    """Streaming per-group mean over cell chunks. Memory = n_groups x n_genes, not n_cells."""

    def __init__(self, genes: Sequence[str]):
        self.genes = list(genes)
        self._sum: Dict[GroupKey, np.ndarray] = {}
        self._count: Dict[GroupKey, int] = {}

    def add(self, obs: pd.DataFrame, X: np.ndarray) -> None:
        """Add a chunk. `obs` has columns pert_id/condition/donor (n rows); `X` is n x n_genes."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != len(self.genes):
            raise ValueError(f"chunk has {X.shape[1]} genes, expected {len(self.genes)}")
        keys = list(zip(obs["pert_id"].astype(str),
                        obs["condition"].astype(str),
                        obs["donor"].astype(str)))
        # group rows of this chunk, accumulate
        order = pd.Series(range(len(keys))).groupby(keys, sort=False)
        for key, rows in order.groups.items():
            r = np.asarray(rows, dtype=int)
            s = X[r].sum(axis=0)
            if key in self._sum:
                self._sum[key] += s
                self._count[key] += len(r)
            else:
                self._sum[key] = s
                self._count[key] = len(r)

    def result(self) -> pd.DataFrame:
        """Mean expression per group: DataFrame indexed by (pert_id, condition, donor)."""
        if not self._sum:
            raise ValueError("no cells accumulated")
        keys = list(self._sum.keys())
        mat = np.vstack([self._sum[k] / self._count[k] for k in keys])
        idx = pd.MultiIndex.from_tuples(keys, names=C.PSEUDOBULK_INDEX_NAMES)
        return pd.DataFrame(mat, index=idx, columns=self.genes).sort_index()

    def counts(self) -> pd.Series:
        keys = list(self._count.keys())
        idx = pd.MultiIndex.from_tuples(keys, names=C.PSEUDOBULK_INDEX_NAMES)
        return pd.Series([self._count[k] for k in keys], index=idx, name="n_cells").sort_index()


def compute_deltas(expr: pd.DataFrame) -> pd.DataFrame:
    """delta[pert,cond,donor] = expr[pert,cond,donor] - expr[control,cond,donor].

    Matched on (condition, donor). A group whose (condition, donor) has no control is
    dropped with a warning (it cannot be turned into a delta). Control rows get delta 0.
    """
    control = expr.xs(C.CONTROL_PERT_ID, level="pert_id", drop_level=True)  # index=(cond,donor)
    out = np.empty_like(expr.to_numpy(dtype=float))
    keep = np.ones(len(expr), dtype=bool)
    for i, (pert, cond, donor) in enumerate(expr.index):
        if (cond, donor) in control.index:
            out[i] = expr.iloc[i].to_numpy() - control.loc[(cond, donor)].to_numpy()
        else:
            keep[i] = False
    if not keep.all():
        import warnings
        warnings.warn(f"{(~keep).sum()} group(s) had no matched control and were dropped")
    delta = pd.DataFrame(out, index=expr.index, columns=expr.columns)
    return delta[keep]


def build_frames(
    expr: pd.DataFrame,
    delta: pd.DataFrame,
    man: C.SplitManifest,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Route each (pert,cond,donor) row into the train or test pseudobulk frame (§3)."""
    expr = expr.loc[delta.index]  # align to rows that survived delta computation
    is_train = np.array([
        split_mod.is_train_row(p, c, d, man) for (p, c, d) in delta.index
    ])
    train = C.build_pseudobulk_frame(expr[is_train], delta[is_train])
    test = C.build_pseudobulk_frame(expr[~is_train], delta[~is_train])
    return train, test


def write(train: pd.DataFrame, test: pd.DataFrame) -> None:
    C.PSEUDOBULK_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(C.PSEUDOBULK_TRAIN)
    test.to_parquet(C.PSEUDOBULK_TEST)


def build_and_write(expr: pd.DataFrame, man: Optional[C.SplitManifest] = None) -> None:
    """Full Lane-C step: means -> deltas -> route -> write both parquets."""
    if man is None:
        man = split_mod.load()
    delta = compute_deltas(expr)
    train, test = build_frames(expr, delta, man)
    write(train, test)


# --- Production data path (backed/chunked AnnData) --------------------------------
def iter_anndata_chunks(h5ad_path, hvg_genes: Sequence[str], chunk: int = 50_000):
    """Yield (obs_chunk, X_chunk) over a backed AnnData, restricted to the HVG columns.

    Lazily imports anndata; never holds more than `chunk` cells dense in memory.
    Assumes obs carries `pert_id`, `condition`, `donor` (normalized by core.data).
    """
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path, backed="r")
    gene_pos = [adata.var_names.get_loc(g) for g in hvg_genes]
    n = adata.n_obs
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sub = adata[start:stop]
        X = sub.X[:, gene_pos]
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        obs = sub.obs[["pert_id", "condition", "donor"]].reset_index(drop=True)
        yield obs, X


def build_from_anndata(h5ad_path, hvg_genes: Sequence[str], man: Optional[C.SplitManifest] = None,
                       chunk: int = 50_000) -> None:
    """Production entrypoint: stream a backed .h5ad into the pseudobulk parquets."""
    acc = PseudobulkAccumulator(hvg_genes)
    for obs, X in iter_anndata_chunks(h5ad_path, hvg_genes, chunk=chunk):
        acc.add(obs, X)
    build_and_write(acc.result(), man)
