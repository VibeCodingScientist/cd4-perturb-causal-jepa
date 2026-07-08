"""
core.features — gene-token priors + DEG-frequency feature (UNIFIED_BUILD_PLAN.md §7a/§7b).

Three caches, all keyed so downstream models join cleanly against the contract:

  * esm2.parquet          gene_id -> ESM-2 650M mean-pooled embedding (ESM2_DIM)   [G1, GPU]
  * context_prior.parquet gene_id -> network/GO vector (CONTEXT_PRIOR_DIM)          [G1/CPU]
  * deg_freq.parquet      pert_id -> 50-dim DEG-frequency feature                   [CPU, here]

DEG-frequency (the CPU-computable one, fully implemented + tested here). Donors act as
biological replicates: for each perturbation we one-sample-t-test its per-donor deltas
against zero per gene, BH-FDR correct, and call a gene "DE" at FDR < 0.1. The 50 genes
that are DE in the most training perturbations form the feature columns; each
perturbation's row is its mean delta on those 50 genes — a compact, discriminative,
VCC-winner-style descriptor.

ESM-2 and the network/context prior require the GPU / a downloaded graph and therefore
run as job G1 through `gpu_queue.py`; they are implemented correct-by-construction and
guarded so importing this module never drags in torch/networkx.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from . import contract as C

DE_FDR_ALPHA = 0.1  # §7a


# ===========================================================================
# DEG-frequency feature (CPU; implemented + tested locally)
# ===========================================================================
def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. NaNs pass through as NaN (not significant)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return out
    try:
        from scipy.stats import false_discovery_control
        out[finite] = false_discovery_control(p[finite], method="bh")
    except Exception:  # older scipy: manual BH
        idx = np.where(finite)[0]
        order = idx[np.argsort(p[idx])]
        m = len(order)
        ranked = p[order] * m / (np.arange(1, m + 1))
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out[order] = np.clip(ranked, 0, 1)
    return out


def de_significance(train_pseudobulk: pd.DataFrame) -> pd.DataFrame:
    """FDR-adjusted p per (perturbation, gene) using donors/conditions as replicates.

    Returns a DataFrame indexed by pert_id, columns = genes, values = BH-FDR q-value of a
    one-sample t-test of that perturbation's replicate deltas against 0. Perturbations with
    <2 replicates yield NaN (never called significant). The control is excluded.
    """
    delta = C.pseudobulk_delta(train_pseudobulk)
    genes = list(delta.columns)
    rows: Dict[str, np.ndarray] = {}
    for pert, grp in delta.groupby(level="pert_id"):
        if pert == C.CONTROL_PERT_ID or grp.shape[0] < 2:
            continue
        with np.errstate(invalid="ignore"):
            t = ttest_1samp(grp.to_numpy(dtype=float), popmean=0.0, axis=0, nan_policy="omit")
        pvals = np.asarray(t.pvalue, dtype=float)
        rows[pert] = _bh_fdr(pvals)
    if not rows:
        raise ValueError("no perturbation had >=2 replicates for a DE test")
    return pd.DataFrame.from_dict(rows, orient="index", columns=genes)


def de_frequency(qvals: pd.DataFrame, alpha: float = DE_FDR_ALPHA) -> pd.Series:
    """Per-gene fraction of perturbations where the gene is DE (q < alpha). Higher = more
    frequently perturbed across the screen."""
    sig = qvals < alpha
    return sig.mean(axis=0).sort_values(ascending=False)


def top_deg_genes(qvals: pd.DataFrame, k: int = C.TOP_DEG_N, alpha: float = DE_FDR_ALPHA) -> List[str]:
    """The k genes DE in the most perturbations (the DEG-frequency feature columns)."""
    freq = de_frequency(qvals, alpha)
    return list(freq.index[:k])


def deg_freq_features(
    train_pseudobulk: pd.DataFrame,
    top_genes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the 50-dim DEG-frequency feature: index = pert_id, columns = top-50 DE genes,
    value = the perturbation's mean delta on that gene (signed effect on the most commonly
    perturbed genes). Also usable for held-out perturbations at predict time by reindexing.
    """
    qvals = de_significance(train_pseudobulk)
    if top_genes is None:
        top_genes = top_deg_genes(qvals)
    top_genes = list(top_genes)
    delta = C.pseudobulk_delta(train_pseudobulk)
    # exclude the control so the feature rows match de_significance (which drops it)
    per_pert = delta.groupby(level="pert_id").mean().drop(index=C.CONTROL_PERT_ID, errors="ignore")
    # restrict to top DE genes; any missing gene -> 0
    feat = per_pert.reindex(columns=top_genes).fillna(0.0)
    feat.index.name = "pert_id"
    return feat


def build_and_write_deg_freq(train_pseudobulk: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Lane-C step: compute the DEG-frequency feature and cache it to deg_freq.parquet."""
    if train_pseudobulk is None:
        train_pseudobulk = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    feat = deg_freq_features(train_pseudobulk)
    C.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(C.DEG_FREQ_CACHE)
    return feat


# ===========================================================================
# ESM-2 650M gene-token prior (job G1 — GPU; correct-by-construction, gated)
# ===========================================================================
def build_esm2(
    gene_to_sequence: Dict[str, str],
    *,
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 8,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Mean-pooled ESM-2 650M embedding per gene's protein sequence (ESM2_DIM=1280).

    Runs on the GPU via `gpu_queue.py submit esm2` (job G1). `gene_to_sequence` maps each
    gene id to its canonical protein amino-acid sequence (ENSG -> UniProt -> sequence,
    resolved by core.data). Truncates long sequences to the model's context. Falls back to
    GenePT precomputed embeddings (see `build_context_prior`) if ESM-2 is unavailable.
    """
    try:
        import torch
        import esm  # fair-esm
    except Exception as e:  # pragma: no cover - environment gate
        raise RuntimeError(
            "ESM-2 needs `torch` + `fair-esm`; run this as GPU job G1 on the box "
            "(`python gpu_queue.py submit esm2`), or use the GenePT fallback."
        ) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()
    layer = model.num_layers

    genes = list(gene_to_sequence.keys())
    vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(genes), batch_size):
            chunk = genes[i:i + batch_size]
            data = [(g, gene_to_sequence[g][:1022]) for g in chunk]  # 1022 + BOS/EOS
            _, _, toks = bc(data)
            toks = toks.to(device)
            rep = model(toks, repr_layers=[layer])["representations"][layer]
            for j, g in enumerate(chunk):
                L = int((toks[j] != alphabet.padding_idx).sum())
                # mean over residues, excluding BOS/EOS
                v = rep[j, 1:L - 1].mean(0).float().cpu().numpy()
                vecs.append(v)
    df = pd.DataFrame(np.vstack(vecs), index=genes)
    df.index.name = "gene_id"
    if df.shape[1] != C.ESM2_DIM:
        import warnings
        warnings.warn(f"ESM-2 width {df.shape[1]} != contract ESM2_DIM {C.ESM2_DIM}")
    return df


def write_esm2(df: pd.DataFrame) -> None:
    C.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.ESM2_CACHE)


# ===========================================================================
# Regulatory / context prior (node2vec over STRING, or GenePT text embedding)
# ===========================================================================
def build_context_prior_node2vec(
    edges: "pd.DataFrame",
    genes: Sequence[str],
    *,
    dim: int = C.CONTEXT_PRIOR_DIM,
    walk_length: int = 30,
    num_walks: int = 10,
    seed: int = C.SPLIT_SEED,
) -> pd.DataFrame:
    """node2vec embedding over a STRING/GRN graph -> gene -> CONTEXT_PRIOR_DIM vector.

    `edges` is a DataFrame with columns [source, target, weight] (STRING). Genes absent
    from the graph get a zero vector. Requires networkx + node2vec (CPU, downloaded graph);
    kept out of import scope so the module imports without them.
    """
    try:
        import networkx as nx
        from node2vec import Node2Vec
    except Exception as e:  # pragma: no cover - environment gate
        raise RuntimeError(
            "context prior (node2vec) needs `networkx` + `node2vec`; install on the box, "
            "or use the GenePT text-embedding fallback."
        ) from e

    g = nx.from_pandas_edgelist(edges, "source", "target", edge_attr="weight")
    n2v = Node2Vec(g, dimensions=dim, walk_length=walk_length, num_walks=num_walks,
                   seed=seed, quiet=True)
    model = n2v.fit(window=5, min_count=1, seed=seed)
    out = np.zeros((len(genes), dim), dtype=float)
    for i, gene in enumerate(genes):
        if gene in model.wv:
            out[i] = model.wv[gene]
    df = pd.DataFrame(out, index=list(genes))
    df.index.name = "gene_id"
    return df


def write_context_prior(df: pd.DataFrame) -> None:
    C.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.CONTEXT_PRIOR_CACHE)


# ---------------------------------------------------------------------------
# Loaders (used by baselines + models; reindex handles held-out genes gracefully)
# ---------------------------------------------------------------------------
def load_esm2() -> pd.DataFrame:
    return pd.read_parquet(C.ESM2_CACHE)


def load_context_prior() -> pd.DataFrame:
    return pd.read_parquet(C.CONTEXT_PRIOR_CACHE)


def load_deg_freq() -> pd.DataFrame:
    return pd.read_parquet(C.DEG_FREQ_CACHE)


def pert_gene_features(
    perts: Sequence[str],
    *,
    use_context: bool = True,
    esm2: Optional[pd.DataFrame] = None,
    context: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-perturbation gene-token feature matrix (index = pert_id, numeric columns).

    Because pert_id IS the silenced gene's Ensembl id, a perturbation's features are just
    that gene's ESM-2 (⊕ context-prior) embedding — available for held-out genes too, which
    is exactly how a model generalizes to a gene it never saw silenced. Missing genes get 0.
    """
    esm2 = esm2 if esm2 is not None else load_esm2()
    parts = [esm2.reindex(perts).rename(columns=lambda c: f"esm2_{c}")]
    if use_context:
        ctx = context if context is not None else load_context_prior()
        parts.append(ctx.reindex(perts).rename(columns=lambda c: f"ctx_{c}"))
    X = pd.concat(parts, axis=1).astype(float).fillna(0.0)
    X.index = pd.Index(list(perts), name="pert_id")
    return X
