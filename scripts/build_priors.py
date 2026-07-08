"""
scripts/build_priors.py — build the gene-token priors for the frozen HVG panel (job G1).

  * esm2.parquet         real ESM-2 650M mean-pooled embedding per HVG gene's protein
                         (sequences fetched via `gget`, which queries Ensembl).
  * context_prior.parquet a PCA projection of the ESM-2 embeddings to CONTEXT_PRIOR_DIM.

Note on the context prior: the plan's intended regulatory/context prior is node2vec over a
STRING graph (core.features.build_context_prior_node2vec, wired for a follow-up once a STRING
edge list + ENSP->ENSG mapping is staged). For CP1 we fill the context slot with a real,
reliable ESM-2-derived projection so the pipeline runs end-to-end; the headline claim (the
do-operator, C1/C2) does not depend on the context prior's provenance. Coverage (fraction of
HVG genes with a fetched protein) is logged so the compromise is explicit.

Run on the box:  ./.venv/bin/python scripts/build_priors.py
"""
import sys

import numpy as np
import pandas as pd

from core import contract as C
from core import features as feat
from core import split as split_mod

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def fetch_sequences(genes):
    """ENSG -> canonical protein sequence via gget. Robust to gget's FASTA-list output and to
    genes that return no protein (they are simply omitted; the caller zero-fills them)."""
    import gget
    hvg_set = set(genes)
    raw = gget.seq(list(genes), translate=True, verbose=False)
    # gget.seq returns a flat FASTA-style list: [">header", "SEQ", ">header", "SEQ", ...]
    seqs = {}
    i = 0
    while i < len(raw) - 1:
        header, seq = str(raw[i]), str(raw[i + 1])
        if header.startswith(">"):
            # find the requested gene id mentioned in the header
            toks = header.lstrip(">").replace(",", " ").split()
            gid = next((t.split(".")[0] for t in toks if t.split(".")[0] in hvg_set), None)
            if gid and seq:
                clean = "".join(ch for ch in seq.upper() if ch in _AA)
                if clean:
                    seqs.setdefault(gid, clean)
            i += 2
        else:
            i += 1
    return seqs


def main():
    hvg = split_mod.load_hvg()
    print(f"[priors] HVG panel: {len(hvg)} genes", flush=True)

    seqs = fetch_sequences(hvg)
    cov = len(seqs) / max(1, len(hvg))
    print(f"[priors] fetched protein sequences for {len(seqs)}/{len(hvg)} genes "
          f"({cov:.1%} coverage)", flush=True)
    if cov < 0.5:
        print("[priors] WARNING: low sequence coverage — check gget/Ensembl connectivity.",
              file=sys.stderr, flush=True)

    esm2 = feat.build_esm2(seqs)                 # (n_covered, ESM2_DIM), GPU
    esm2 = esm2.reindex(hvg).fillna(0.0)          # align to HVG; unmapped -> zero prior
    esm2.index.name = "gene_id"
    feat.write_esm2(esm2)
    print(f"[priors] wrote esm2.parquet {esm2.shape}", flush=True)

    # Context prior = PCA projection of ESM-2 to CONTEXT_PRIOR_DIM (real; see module docstring).
    from sklearn.decomposition import PCA
    X = esm2.to_numpy()
    k = min(C.CONTEXT_PRIOR_DIM, X.shape[1], X.shape[0])
    proj = PCA(n_components=k, random_state=C.SPLIT_SEED).fit_transform(X)
    if proj.shape[1] < C.CONTEXT_PRIOR_DIM:
        proj = np.pad(proj, ((0, 0), (0, C.CONTEXT_PRIOR_DIM - proj.shape[1])))
    ctx = pd.DataFrame(proj, index=hvg)
    ctx.index.name = "gene_id"
    feat.write_context_prior(ctx)
    print(f"[priors] wrote context_prior.parquet {ctx.shape}", flush=True)


if __name__ == "__main__":
    main()
