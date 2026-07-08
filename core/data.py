"""
core.data — dataset ingestion for GSE278572 (UNIFIED_BUILD_PLAN.md §7a).

Backed / chunked h5ad only: the raw object is ~22M cells and must NEVER be densified in
full (64 GB RAM). This module normalizes obs to the canonical schema, runs QC, selects
3,000 HVGs on a bounded stratified subsample, and exposes a stratified cell sampler that
the JEPA lane consumes.

Canonical obs schema (produced by `normalize_obs`):
  * pert_id   — Ensembl gene id of the silenced gene, or contract.CONTROL_PERT_ID for NTC.
  * condition — one of contract.CONDITIONS (Rest / Stim8hr / Stim48hr).
  * donor     — one of contract.DONORS (donor_1..donor_4).
Canonical var: var_names are Ensembl gene ids (contract.GENE_ID == "ensembl").

anndata / scanpy are imported lazily so `import core.data` never requires them; the heavy
steps run on the box. The pipeline orchestrator `prepare_core()` chains
data -> split.freeze -> pseudobulk -> DEG-frequency for the CP1 Lane-C build.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import contract as C

# Column-name candidates seen across GEO/CZI mirrors of this dataset; normalize_obs maps
# whichever is present onto the canonical names above.
_PERT_COLS = ("gene", "target_gene", "perturbation", "guide_target", "gene_target", "KO")
_COND_COLS = ("condition", "activation", "stim", "state", "timepoint", "activation_state")
_DONOR_COLS = ("donor", "donor_id", "individual", "patient", "sample_donor")
_CONTROL_TOKENS = ("non-targeting", "nt", "ntc", "control", "safe-harbor", "no-target", "ctrl")


def read_backed(h5ad_path):
    """Open the dataset in backed mode (no dense materialization)."""
    import anndata as ad
    return ad.read_h5ad(h5ad_path, backed="r")


# ---------------------------------------------------------------------------
# obs / var normalization
# ---------------------------------------------------------------------------
def _first_present(cols: Sequence[str], available) -> Optional[str]:
    lower = {c.lower(): c for c in available}
    for cand in cols:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _canon_condition(v: str) -> str:
    s = str(v).lower().replace(" ", "").replace("_", "").replace("-", "")
    if s.startswith("rest") or s in ("0h", "unstim", "unstimulated"):
        return "Rest"
    if "48" in s:
        return "Stim48hr"
    if "8" in s:
        return "Stim8hr"
    # already canonical?
    for c in C.CONDITIONS:
        if s == c.lower():
            return c
    return str(v)


def _canon_donor(v: str) -> str:
    s = str(v)
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"donor_{digits}" if digits else s


def _canon_pert(v: str, symbol_to_ensembl: Optional[Dict[str, str]] = None) -> str:
    s = str(v).strip()
    if s.lower() in _CONTROL_TOKENS or any(tok in s.lower() for tok in ("non-target", "nontarget")):
        return C.CONTROL_PERT_ID
    if s.startswith("ENSG"):
        return s
    if symbol_to_ensembl and s in symbol_to_ensembl:
        return symbol_to_ensembl[s]
    return s  # leave as-is; caller may map later


def normalize_obs(adata, symbol_to_ensembl: Optional[Dict[str, str]] = None) -> None:
    """Rewrite adata.obs in place to carry canonical pert_id / condition / donor columns."""
    obs = adata.obs
    pcol = _first_present(_PERT_COLS, obs.columns)
    ccol = _first_present(_COND_COLS, obs.columns)
    dcol = _first_present(_DONOR_COLS, obs.columns)
    missing = [n for n, c in (("pert", pcol), ("condition", ccol), ("donor", dcol)) if c is None]
    if missing:
        raise ValueError(
            f"could not locate obs columns for {missing}; available: {list(obs.columns)}. "
            "Pass the right column via a pre-rename or extend the _*_COLS candidates."
        )
    adata.obs["pert_id"] = [_canon_pert(v, symbol_to_ensembl) for v in obs[pcol]]
    adata.obs["condition"] = [_canon_condition(v) for v in obs[ccol]]
    adata.obs["donor"] = [_canon_donor(v) for v in obs[dcol]]


def ensure_ensembl_var(adata, symbol_to_ensembl: Optional[Dict[str, str]] = None) -> None:
    """Ensure var_names are Ensembl ids. If they are symbols and a map is given, translate;
    drop genes with no mapping. No-op if already ENSG."""
    if all(str(v).startswith("ENSG") for v in adata.var_names[:20]):
        return
    if not symbol_to_ensembl:
        raise ValueError("var_names are not Ensembl and no symbol->ENSG map was provided")
    keep = [v in symbol_to_ensembl for v in adata.var_names]
    adata._inplace_subset_var(np.asarray(keep))
    adata.var_names = [symbol_to_ensembl[v] for v in adata.var_names]


# ---------------------------------------------------------------------------
# QC + normalization + HVG
# ---------------------------------------------------------------------------
def qc_and_normalize(adata, *, min_genes: int = 200, min_cells: int = 3,
                     max_pct_mt: float = 15.0, target_sum: float = 1e4):
    """Standard QC + log1p-CP10k (§7a). Operates on an in-memory (sub)set; the caller feeds
    it a subsample for HVG selection, not the full 22M cells."""
    import scanpy as sc
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if "mt" not in adata.var:
        adata.var["mt"] = [str(n).startswith(("MT-", "mt-")) for n in adata.var_names]
    if adata.var["mt"].any():
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_hvg(adata_subsample, n_top: int = C.HVG_N) -> List[str]:
    """Select `n_top` HVGs on a normalized subsample; return Ensembl ids in a stable order."""
    import scanpy as sc
    a = adata_subsample.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat")
    hvg = list(a.var_names[a.var["highly_variable"].to_numpy()])
    return hvg[:n_top]


# ---------------------------------------------------------------------------
# Stratified cell sampling (HVG selection + JEPA lane)
# ---------------------------------------------------------------------------
def stratified_cell_indices(obs: pd.DataFrame, n: int, *, seed: int = C.SPLIT_SEED,
                            strata=("condition", "donor")) -> np.ndarray:
    """Row indices of a size-`n` sample stratified over `strata` (proportional allocation)."""
    rng = np.random.default_rng(seed)
    groups = obs.groupby(list(strata), sort=False).indices
    total = len(obs)
    picks: List[int] = []
    for _, idx in groups.items():
        k = max(1, int(round(n * len(idx) / total)))
        k = min(k, len(idx))
        picks.extend(rng.choice(idx, size=k, replace=False).tolist())
    picks = np.array(sorted(set(picks)))
    if len(picks) > n:
        picks = np.array(sorted(rng.choice(picks, size=n, replace=False)))
    return picks


def perturbed_genes(adata) -> List[str]:
    """Every silenced gene (the gene-holdout universe), excluding the control."""
    perts = pd.unique(adata.obs["pert_id"])
    return sorted(str(p) for p in perts if p != C.CONTROL_PERT_ID)


# ---------------------------------------------------------------------------
# Orchestration (box entrypoint for the CP1 Lane-C core build)
# ---------------------------------------------------------------------------
def prepare_core(
    h5ad_path,
    *,
    doi: str = "",
    symbol_to_ensembl: Optional[Dict[str, str]] = None,
    hvg_subsample: int = 200_000,
) -> None:
    """End-to-end Lane-C core build on the box:
    normalize -> QC(subsample) -> HVG -> split.freeze -> pseudobulk -> DEG-frequency.

    Never densifies the full matrix: HVG is chosen on a stratified subsample; pseudobulk is
    streamed chunk-by-chunk over the backed object.
    """
    from . import split as split_mod
    from . import pseudobulk as pb
    from . import features as feat

    adata = read_backed(h5ad_path)
    normalize_obs(adata, symbol_to_ensembl)
    ensure_ensembl_var(adata, symbol_to_ensembl)

    # HVG on a bounded stratified subsample (loaded to memory just for these cells)
    sub_idx = stratified_cell_indices(adata.obs, hvg_subsample)
    sub = adata[sub_idx].to_memory()
    sub = qc_and_normalize(sub)
    hvg = select_hvg(sub, C.HVG_N)

    # Freeze the split against this exact file
    genes = perturbed_genes(adata)
    man = split_mod.freeze(perturbed_genes=genes, hvg_genes=hvg, h5ad_path=Path(h5ad_path), doi=doi)

    # Pseudobulk (streamed) + DEG-frequency
    pb.build_from_anndata(h5ad_path, hvg, man)
    feat.build_and_write_deg_freq()
