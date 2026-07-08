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


def _frac_ensembl(names) -> float:
    s = [str(v) for v in list(names)[:500]]
    return sum(v.startswith("ENSG") for v in s) / max(1, len(s))


def ensure_ensembl_var(adata, symbol_to_ensembl: Optional[Dict[str, str]] = None) -> None:
    """Ensure var_names are Ensembl ids. No-op if they already are (a handful of custom
    spike-ins like PuroR are tolerated). Otherwise use the `gene_ids` column, or RENAME symbols
    via the provided map (unmapped kept as-is; HVG selection later keeps only ENSG names).
    Rename-only avoids a fragile inplace var subset on a backed AnnData."""
    if _frac_ensembl(adata.var_names) >= 0.8:
        return
    if "gene_ids" in adata.var.columns and _frac_ensembl(adata.var["gene_ids"]) >= 0.8:
        adata.var_names = list(adata.var["gene_ids"])
    elif symbol_to_ensembl:
        adata.var_names = [symbol_to_ensembl.get(str(v), str(v)) for v in adata.var_names]
    else:
        raise ValueError("var_names are not Ensembl and no symbol->ENSG map or gene_ids column")
    adata.var_names_make_unique()


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
# CZI pre-computed pseudobulk adapter (the CP1 data path)
# ---------------------------------------------------------------------------
# The CZI Virtual Cells mirror ships `GWCD4i.pseudobulk_merged.h5ad` (44.6 GB): one row per
# (guide, donor, culture_condition), X = summed UMI counts over 18,129 genes. CP1 (baselines +
# causal) runs on pseudobulk deltas, so we adapt this file directly instead of streaming the
# ~1.7 TB of single cells (that is the JEPA lane's job). obs columns are documented in the
# dataset's data_sharing_readme.md.
def czi_obs_to_canonical(obs: pd.DataFrame,
                         donor_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Map CZI pseudobulk obs -> canonical (pert_id, condition, donor).

    pert_id = perturbed_gene_id (Ensembl) for targeting guides, CONTROL_PERT_ID for
    non-targeting. Donor codes (e.g. CE0008162) are relabeled donor_1..N via `donor_map` so the
    donor probe (donor_4) resolves; without a map they fall back to a digit-derived label.
    Pure/testable (no anndata)."""
    gtype = obs["guide_type"].astype(str).str.lower()
    is_ntc = gtype.str.contains("non") & gtype.str.contains("target")
    gid = obs["perturbed_gene_id"].astype(str)
    pert = [C.CONTROL_PERT_ID if ntc else g for ntc, g in zip(is_ntc, gid)]
    donor_map = donor_map or {}
    donor = [donor_map.get(str(d), _canon_donor(d)) for d in obs["donor_id"]]
    return pd.DataFrame(
        {"pert_id": pert,
         "condition": [_canon_condition(v) for v in obs["culture_condition"]],
         "donor": donor},
        index=obs.index,
    )


def czi_donor_map(obs: pd.DataFrame) -> Dict[str, str]:
    """Deterministic donor-code -> donor_1..N map (sorted), so donor_4 is a stable probe."""
    codes = sorted(str(d) for d in pd.unique(obs["donor_id"]))
    return {c: f"donor_{i + 1}" for i, c in enumerate(codes)}


def normalize_pseudobulk_counts(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Summed-UMI-count pseudobulk -> log1p CP10k, per row. Pure/testable."""
    X = np.asarray(X, dtype=np.float64)
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(X / tot * target_sum)


def _czi_quality_mask(obs: pd.DataFrame) -> np.ndarray:
    """Keep quality pseudobulks (targeting rows via keep_for_DE; controls via keep_min_cells)."""
    n = len(obs)
    if "keep_min_cells" not in obs.columns:
        return np.ones(n, dtype=bool)
    keep_min = obs["keep_min_cells"].to_numpy(dtype=bool)
    keep_total = obs["keep_total_counts"].to_numpy(dtype=bool) if "keep_total_counts" in obs else np.ones(n, bool)
    base = keep_min & keep_total
    gtype = obs["guide_type"].astype(str).str.lower()
    is_ntc = (gtype.str.contains("non") & gtype.str.contains("target")).to_numpy()
    if "keep_for_DE" in obs.columns:
        keep_de = obs["keep_for_DE"].to_numpy(dtype=bool)
        return np.where(is_ntc, base, base & keep_de)
    return base


def build_from_czi_pseudobulk(
    h5ad_path,
    *,
    doi: str = "",
    hvg_subsample: int = 20_000,
    chunk: int = 20_000,
    quality_filter: bool = True,
) -> None:
    """CP1 Lane-C build from the CZI pseudobulk h5ad: normalize per-guide profiles, average
    guides -> per (gene, condition, donor), select HVG, freeze the split, write pseudobulk +
    DEG-frequency. Streams in chunks (never densifies the whole 44 GB file at once).
    """
    from pathlib import Path
    import anndata as ad
    from . import split as split_mod
    from . import pseudobulk as pb
    from . import features as feat

    adata = ad.read_h5ad(h5ad_path, backed="r")
    ensure_ensembl_var(adata)  # var_names already Ensembl (tolerates custom spike-ins)
    genes_all = list(adata.var_names)
    n = adata.n_obs
    donor_map = czi_donor_map(adata.obs)
    qmask_all = _czi_quality_mask(adata.obs) if quality_filter else np.ones(n, dtype=bool)

    # HVG on a normalized subsample of quality pseudobulks, restricted to real ENSG genes
    ens_mask = np.array([str(v).startswith("ENSG") for v in genes_all])
    keep_pos = np.where(qmask_all)[0]
    rng = np.random.default_rng(C.SPLIT_SEED)
    sub_pos = np.sort(rng.choice(keep_pos, size=min(hvg_subsample, len(keep_pos)), replace=False))
    sub = adata[sub_pos].to_memory()
    sub = sub[:, ens_mask].copy()
    subX = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    import scanpy as sc
    sub.X = normalize_pseudobulk_counts(subX)
    sc.pp.highly_variable_genes(sub, n_top_genes=C.HVG_N, flavor="seurat")
    hvg = list(sub.var_names[sub.var["highly_variable"].to_numpy()])[:C.HVG_N]
    gene_pos = [genes_all.index(g) for g in hvg]

    # Stream: average NORMALIZED guide-profiles into per (pert,cond,donor) pseudobulk
    acc = pb.PseudobulkAccumulator(hvg)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        m = qmask_all[start:stop]
        if not m.any():
            continue
        sl = adata[start:stop]
        Xc = sl.X[:, gene_pos]
        Xc = Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)
        Xc = normalize_pseudobulk_counts(Xc)[m]
        obs_c = czi_obs_to_canonical(sl.obs, donor_map).reset_index(drop=True)[m]
        acc.add(obs_c, Xc)

    expr = acc.result()
    perts = sorted(set(expr.index.get_level_values("pert_id")) - {C.CONTROL_PERT_ID})
    man = split_mod.freeze(perturbed_genes=perts, hvg_genes=hvg, h5ad_path=Path(h5ad_path), doi=doi)
    pb.build_and_write(expr, man)
    feat.build_and_write_deg_freq()


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

    # Pseudobulk (streamed from the NORMALIZED in-memory object) + DEG-frequency
    pb.build_from_anndata(adata, hvg, man)
    feat.build_and_write_deg_freq()
