"""
core.synthetic — a small, dependency-light synthetic dataset for exercising the whole
Lane-C core (pseudobulk deltas, split routing, DEG-frequency, eval, baselines) WITHOUT
torch / anndata / the real 22M-cell download.

It uses a latent linear-causal generative model so there is genuine learnable structure:
knocking down gene k reduces k and propagates one hop through a sparse regulatory matrix
W, with a condition-dependent gain (so the Stim48hr condition hold-out is a real shift).
This is what the `core-frozen` gate and CI run against; it is NOT scientific data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import contract as C
from . import split as split_mod


def _gene_ids(n: int) -> List[str]:
    return [f"ENSG{i:08d}" for i in range(n)]


def _generate(
    *,
    n_genes: int = 200,
    n_perts: int = 120,
    n_donors: int = 4,
    seed: int = 0,
) -> Tuple[pd.DataFrame, dict]:
    """Return (group_expr, latents). `latents["prop"][:, k]` is the propagated effect of
    knocking down gene k — used to build informative synthetic ESM-2 embeddings so the
    baselines have genuine learnable signal (not just a plumbing check)."""
    rng = np.random.default_rng(seed)
    genes = _gene_ids(n_genes)
    perturbed = genes[:n_perts]  # each perturbation silences one of these genes
    donors = [f"donor_{i+1}" for i in range(n_donors)]

    # sparse regulatory matrix W (targets x regulators); knockdown of k -> -(W+0.3 W^2)[:,k]
    W = np.zeros((n_genes, n_genes))
    n_edges = n_genes * 4
    src = rng.integers(0, n_genes, n_edges)
    dst = rng.integers(0, n_genes, n_edges)
    W[dst, src] = rng.normal(0, 1.0, n_edges)
    np.fill_diagonal(W, 1.0)  # a gene strongly regulates itself (target of its own knockdown)
    prop = W + 0.3 * (W @ W)

    baseline = rng.gamma(shape=2.0, scale=1.0, size=n_genes)  # positive absolute levels
    cond_gain = {"Rest": 1.0, "Stim8hr": 1.25, "Stim48hr": 1.7}  # Stim48hr = the shift
    # small condition-specific twist so the hold-out is not a pure rescale
    cond_twist = {c: rng.normal(0, 0.15, (n_genes, n_genes)) for c in C.CONDITIONS}

    rows: Dict[Tuple[str, str, str], np.ndarray] = {}
    for cond in C.CONDITIONS:
        Mc = cond_gain[cond] * prop + cond_twist[cond]
        for donor in donors:
            donor_eff = rng.normal(0, 0.05, n_genes)
            # control
            rows[(C.CONTROL_PERT_ID, cond, donor)] = (
                baseline + donor_eff + rng.normal(0, 0.02, n_genes)
            )
            # perturbations
            for k, gk in enumerate(perturbed):
                delta = -Mc[:, k]  # knockdown of gene k, propagated one hop
                expr = baseline + donor_eff + delta + rng.normal(0, 0.05, n_genes)
                rows[(gk, cond, donor)] = expr

    idx = pd.MultiIndex.from_tuples(list(rows.keys()), names=C.PSEUDOBULK_INDEX_NAMES)
    expr = pd.DataFrame(np.vstack(list(rows.values())), index=idx, columns=genes).sort_index()
    latents = {"prop": prop, "genes": genes, "perturbed": perturbed, "n_genes": n_genes}
    return expr, latents


def make_group_expr(*, n_genes: int = 200, n_perts: int = 120, n_donors: int = 4,
                    seed: int = 0) -> pd.DataFrame:
    """Group-level (pert_id, condition, donor) mean expression with real causal structure."""
    expr, _ = _generate(n_genes=n_genes, n_perts=n_perts, n_donors=n_donors, seed=seed)
    return expr


def write_synthetic(
    *,
    n_genes: int = 200,
    n_perts: int = 120,
    seed: int = 0,
    freeze_split: bool = True,
) -> Dict[str, object]:
    """Materialize a full synthetic core into the current DATA_ROOT + repo split files.

    Writes: split_manifest.json (data-frozen), split/hvg_3000-analog list, pseudobulk
    train/test parquets, esm2 / context_prior / deg_freq caches. Returns key artifacts for
    assertions. Set CD4_DATA_ROOT to a temp dir BEFORE importing core.contract to sandbox.
    """
    from . import pseudobulk as pb
    from . import features as feat

    C.ensure_dirs()
    genes = _gene_ids(n_genes)
    perturbed = genes[:n_perts]

    expr, latents = _generate(n_genes=n_genes, n_perts=n_perts, seed=seed)
    prop = latents["prop"]  # prop[:, k] = propagated effect of knocking down gene k

    if freeze_split:
        # HVG "list" = all synthetic genes (n_genes stands in for HVG_N here). Suppress the
        # expected count-mismatch warning since the synthetic gene universe is intentionally small.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="hvg_genes has")
            man = split_mod.freeze(
                perturbed_genes=perturbed, hvg_genes=genes, h5ad_path=None,
                doi="synthetic", write=True,
            )
    else:
        man = split_mod.load()

    # Real pseudobulk code path: deltas + train/test routing
    pb.build_and_write(expr, man)

    # DEG-frequency via the real features code path
    train_pb = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    feat.build_and_write_deg_freq(train_pb)

    # Feature caches with correct contract dims. To give baselines genuine learnable signal,
    # a perturbed gene's ESM-2 stand-in encodes its latent effect prop[:, k] in the leading
    # dims (+ noise); everything else is random. This mirrors "the gene embedding carries the
    # information needed to predict the knockdown response", so gene hold-out is non-trivial.
    rng = np.random.default_rng(seed + 1)
    esm2_mat = rng.normal(0, 1.0, (n_genes, C.ESM2_DIM))
    lead = min(n_genes, C.ESM2_DIM)
    for k, gk in enumerate(perturbed):
        gi = genes.index(gk)
        esm2_mat[gi, :lead] = prop[:lead, k] + rng.normal(0, 0.3, lead)
    esm2 = pd.DataFrame(esm2_mat, index=genes); esm2.index.name = "gene_id"
    feat.write_esm2(esm2)
    # context prior: node2vec-analog carrying each gene's regulator row (+ noise)
    ctx_mat = rng.normal(0, 1.0, (n_genes, C.CONTEXT_PRIOR_DIM))
    lead_c = min(n_genes, C.CONTEXT_PRIOR_DIM)
    for k, gk in enumerate(perturbed):
        gi = genes.index(gk)
        ctx_mat[gi, :lead_c] = prop[k, :lead_c] + rng.normal(0, 0.3, lead_c)
    ctx = pd.DataFrame(ctx_mat, index=genes); ctx.index.name = "gene_id"
    feat.write_context_prior(ctx)

    return {
        "manifest": man,
        "genes": genes,
        "perturbed": perturbed,
        "expr": expr,
        "train": train_pb,
        "test": pd.read_parquet(C.PSEUDOBULK_TEST),
    }


if __name__ == "__main__":  # pragma: no cover
    art = write_synthetic()
    print("wrote synthetic core to", C.DATA_ROOT)
    print("  train rows:", len(art["train"]), " test rows:", len(art["test"]))
    print("  gene_holdout:", len(art["manifest"].gene_holdout))
