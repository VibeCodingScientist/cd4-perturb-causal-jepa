#!/usr/bin/env python
"""C-BUDGET Stage 1 — the noise floor (Bucket B) + shared-vs-specific reliability.

CPU-only. Reads the committed CZI guide-level pseudobulk (GWCD4i.pseudobulk_merged.h5ad;
one row per guide x donor x condition) + the frozen split, in the SAME normalization/
control space as the frozen eval (core.data.normalize_pseudobulk_counts). For each
evaluable perturbation it estimates split-half reliability over (guide, donor) replicate
units, disattenuates to the full-sample ceiling via Spearman-Brown, and brackets it with
within-donor (technical) and cross-donor (biological) floors. Also decomposes each
response into a shared (mean-across-perturbations) component and a perturbation-specific
residual, and reports the reliability of each — the number that explains mode collapse.

Writes results/budget_reliability.csv. Never touches CP2 or Dev 4's artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import contract as C
from core import data as d1data
from core import eval as ev
from core import split as split_mod

CZI = C.RAW_DIR / "GWCD4i.pseudobulk_merged.h5ad"
OUT = C.RESULTS_DIR / "budget_reliability.csv"
N_SPLITS = 20
SEED = 42
TOPK = C.TOP_DEG_N  # 50


def _spearman_brown(r_half: float) -> float:
    if not np.isfinite(r_half):
        return np.nan
    r = max(-0.999, min(0.999, r_half))
    return 2.0 * r / (1.0 + r)


def _corr_over(a: np.ndarray, b: np.ndarray, cols: np.ndarray) -> float:
    x, y = a[cols], b[cols]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def load_czi_deltas(hvg, evaluable_genes, conditions):
    """Return per-guide deltas in the frozen space, restricted to the needed genes/conditions.

    Structure: deltas[(pert_id, condition)] = list of (donor, guide_delta_vector[HVG]).
    Delta = normalized(guide profile) - matched control (mean of non-targeting at donor,cond).
    """
    import anndata as ad

    adata = ad.read_h5ad(CZI, backed="r")
    d1data.ensure_ensembl_var(adata)
    genes_all = list(adata.var_names)
    gpos = np.array([genes_all.index(g) for g in hvg])  # HVG column positions
    obs = adata.obs
    canon = d1data.czi_obs_to_canonical(obs, d1data.czi_donor_map(obs))
    qmask = d1data._czi_quality_mask(obs)
    cond = canon["condition"].to_numpy()
    pert = canon["pert_id"].to_numpy()
    donor = canon["donor"].to_numpy()

    want = np.isin(cond, list(conditions)) & qmask
    is_ctrl = pert == C.CONTROL_PERT_ID
    is_eval = np.isin(pert, list(evaluable_genes))
    rows = np.flatnonzero(want & (is_ctrl | is_eval))
    print(f"[czi] reading {len(rows)} rows (of {adata.n_obs}) on {len(hvg)} HVG ...", flush=True)

    # read the needed rows' HVG columns in chunks (sparse -> dense per chunk), normalize
    prof = np.empty((len(rows), len(hvg)), dtype=np.float32)
    for i0 in range(0, len(rows), 8000):
        sl = np.sort(rows[i0:i0 + 8000])
        X = adata[sl].to_memory().X
        X = X[:, gpos]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        prof[i0:i0 + len(sl)] = d1data.normalize_pseudobulk_counts(X).astype(np.float32)
    rp, rc, rd, ri = pert[rows], cond[rows], donor[rows], is_ctrl[rows]

    # matched controls: mean non-targeting profile per (condition, donor)
    ctrl = {}
    for c in conditions:
        for dn in np.unique(rd):
            m = (rc == c) & (rd == dn) & ri
            if m.any():
                ctrl[(c, dn)] = prof[m].mean(0)

    deltas = {}
    for j in range(len(rows)):
        if ri[j]:
            continue
        key = (rp[j], rc[j])
        cb = ctrl.get((rc[j], rd[j]))
        if cb is None:
            continue
        deltas.setdefault(key, []).append((rd[j], prof[j] - cb))
    return deltas


def split_half_reliability(units, cols, rng, n_splits=N_SPLITS):
    """units: list of effect vectors (replicate estimates). Random-split-half corr on `cols`,
    averaged over n_splits, then Spearman-Brown to the full-sample ceiling."""
    k = len(units)
    if k < 4:
        return np.nan, np.nan
    U = np.stack(units)
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(k)
        h1, h2 = perm[: k // 2], perm[k // 2:]
        rs.append(_corr_over(U[h1].mean(0), U[h2].mean(0), cols))
    r_half = float(np.nanmean(rs))
    return r_half, _spearman_brown(r_half)


def main():
    C.ensure_dirs()
    rng = np.random.default_rng(SEED)
    hvg = split_mod.load_hvg()
    gene_ids = np.array(hvg)

    rows_out = []
    for split, conds in [(C.SPLIT_CONDITION, (C.CONDITION_HOLDOUT,)), (C.SPLIT_GENE, C.TRAIN_CONDITIONS)]:
        truth = ev.ground_truth(split)                       # per-pert true delta (frozen)
        perts = [p for p in ev.evaluable_perts(split) if p in truth.index]
        truth = truth.loc[perts]
        deltas = load_czi_deltas(hvg, set(perts), conds)

        # shared component per split (mean true delta across perturbations)
        shared = truth.to_numpy().mean(0)

        n_ok = 0
        for p in perts:
            # gather replicate units across the split's conditions
            units = []
            for c in conds:
                units += [v for (_dn, v) in deltas.get((p, c), [])]
            if len(units) < 4:
                continue
            t = truth.loc[p].to_numpy()
            topk = np.argsort(-np.abs(t))[:TOPK]              # this pert's top-50 true DEGs

            r_half, r_ceil = split_half_reliability(units, topk, rng)
            # specific residual reliability: subtract the shared program from each unit
            spec_units = [u - shared for u in units]
            _, r_ceil_spec = split_half_reliability(spec_units, topk, rng)

            # cross-donor (biological) vs within-donor (technical) brackets
            by_donor = {}
            for c in conds:
                for dn, v in deltas.get((p, c), []):
                    by_donor.setdefault(dn, []).append(v)
            donor_means = {dn: np.mean(vs, 0) for dn, vs in by_donor.items() if len(vs) >= 1}
            cross = [
                _corr_over(donor_means[a], donor_means[b], topk)
                for i, a in enumerate(sorted(donor_means)) for b in sorted(donor_means)[i + 1:]
            ]
            cross_r = float(np.nanmean(cross)) if cross else np.nan
            within = [
                split_half_reliability(vs, topk, rng)[0]
                for vs in by_donor.values() if len(vs) >= 4
            ]
            within_r = float(np.nanmean(within)) if within else np.nan

            rows_out.append({
                "pert_id": p, "split": split, "n_units": len(units),
                "signal_l2": float(np.linalg.norm(t[topk])),
                "r_half": r_half, "r_ceiling": r_ceil, "noise_floor": 1 - r_ceil if np.isfinite(r_ceil) else np.nan,
                "r_ceiling_specific": r_ceil_spec,
                "cross_donor_r": cross_r, "within_donor_r": within_r,
            })
            n_ok += 1
        print(f"[{split}] reliability computed for {n_ok}/{len(perts)} perturbations", flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT, index=False)
    print(f"\n=== wrote {OUT} ({len(df)} rows) ===")
    for split in df["split"].unique():
        s = df[df["split"] == split]
        print(f"\n[{split}] median r_ceiling(full)={s.r_ceiling.median():.3f}  "
              f"noise_floor={s.noise_floor.median():.3f}  "
              f"r_ceiling_specific={s.r_ceiling_specific.median():.3f}  "
              f"cross_donor_r={s.cross_donor_r.median():.3f}  within_donor_r={s.within_donor_r.median():.3f}  n={len(s)}")


if __name__ == "__main__":
    main()
