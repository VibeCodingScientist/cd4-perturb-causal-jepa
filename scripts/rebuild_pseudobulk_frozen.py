#!/usr/bin/env python
"""Rebuild pseudobulk (train/test) + deg_freq from the CZI, PINNED to the committed frozen HVG +
committed frozen split — WITHOUT recomputing HVG or re-freezing (never overwrites the committed
hvg_3000.txt / split_manifest.json).

Why: the box's DATA_ROOT pseudobulk was clobbered by synthetic test fixtures. `build_from_czi_pseudobulk`
would recompute HVG on a random subsample and unconditionally overwrite the frozen panel — risking a
subtly different HVG that desyncs the ESM2 features and the committed measurable-edge table. This variant
reuses the exact accumulator/normalization but keeps the panel + split byte-identical to the frozen
submission. Idempotent: safe to re-run.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import anndata as ad

from core import contract as C
from core import data as D
from core import split as split_mod
from core import pseudobulk as pb
from core import features as feat

CZI = os.environ.get("CD4_CZI", str(C.RAW_DIR / "GWCD4i.pseudobulk_merged.h5ad"))


def main():
    chunk = 20_000
    C.ensure_dirs()
    print(f"[rebuild] CZI={CZI}", flush=True)
    adata = ad.read_h5ad(CZI, backed="r")
    D.ensure_ensembl_var(adata)
    genes_all = list(adata.var_names)
    n = adata.n_obs

    # committed frozen panel + split — NOT recomputed
    hvg = split_mod.load_hvg()                       # 3000 committed stripped-ENSG genes, in order
    man = split_mod.load()                           # committed frozen manifest (perts, gene_holdout, sha)
    print(f"[rebuild] committed HVG={len(hvg)}  gene_holdout={len(man.gene_holdout)}  perts={len(man.perturbed_genes)}", flush=True)

    strip = lambda s: str(s).split(".")[0]
    pos = {}
    for i, g in enumerate(genes_all):
        pos.setdefault(strip(g), i)
    missing = [g for g in hvg if strip(g) not in pos]
    if missing:
        raise SystemExit(f"[rebuild] FATAL: {len(missing)} committed HVG genes absent from CZI var "
                         f"(e.g. {missing[:5]}) — panel/CZI mismatch, refusing to build")
    gene_pos = [pos[strip(g)] for g in hvg]          # CZI column index per committed HVG gene, in HVG order

    donor_map = D.czi_donor_map(adata.obs)
    qmask_all = D._czi_quality_mask(adata.obs)
    print(f"[rebuild] n_obs={n}  quality-pass={int(qmask_all.sum())}  streaming chunk={chunk}", flush=True)

    acc = pb.PseudobulkAccumulator(hvg)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        m = qmask_all[start:stop]
        if not m.any():
            continue
        sl = adata[start:stop]
        Xc = sl.X[:, gene_pos]
        Xc = Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)
        Xc = D.normalize_pseudobulk_counts(Xc)[m]
        obs_c = D.czi_obs_to_canonical(sl.obs, donor_map).reset_index(drop=True)[m]
        acc.add(obs_c, Xc)
        if start % (chunk * 25) == 0:
            print(f"[rebuild]   ...{stop}/{n}", flush=True)

    expr = acc.result()
    got = sorted(set(expr.index.get_level_values("pert_id")) - {C.CONTROL_PERT_ID})
    print(f"[rebuild] accumulated perts={len(got)}  writing pseudobulk with committed manifest", flush=True)
    pb.build_and_write(expr, man)                    # splits train/test by committed gene_holdout
    feat.build_and_write_deg_freq()
    print("REBUILD_PSEUDOBULK_DONE", flush=True)


if __name__ == "__main__":
    main()
