#!/usr/bin/env python
"""C-BUDGET Stage 2 — fraction-of-ceiling (benchmark reframing) + the cross-donor
permutation null (is the perturbation-specific signal real, past chance?).

CPU-only. Consumes results/budget_reliability.csv (Stage 1) + the committed runs/ +
the CZI donor structure. Writes results/fraction_of_ceiling.csv and
results/budget_cross_donor.csv. Never touches CP2 or Dev 4's artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import contract as C
from core import eval as ev
from core import split as split_mod
from budget_reliability import load_czi_deltas, _corr_over

MODELS = [C.MODEL_RIDGE, C.MODEL_NONCAUSAL, C.MODEL_CAUSAL, C.MODEL_JEPA_ONLY, C.MODEL_JEPA_CAUSAL, C.MODEL_FCN]
TOPK = C.TOP_DEG_N
REL = C.RESULTS_DIR / "budget_reliability.csv"
SEED = 42


def per_pert_pearson(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """corr(pred_i, true_i) over each perturbation's own top-50 true DEGs."""
    genes = [g for g in truth.columns if g in pred.columns]
    perts = [p for p in truth.index if p in pred.index]
    P = pred.loc[perts, genes].to_numpy(float)
    T = truth.loc[perts, genes].to_numpy(float)
    out = {}
    for i, p in enumerate(perts):
        topk = np.argsort(-np.abs(T[i]))[:TOPK]
        out[p] = _corr_over(P[i], T[i], topk)
    return out


def cluster_bootstrap_median(vals, n=2000, seed=SEED):
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fraction_of_ceiling(rel: pd.DataFrame):
    rows = []
    for split in [C.SPLIT_CONDITION, C.SPLIT_GENE]:
        truth = ev.ground_truth(split)
        rsub = rel[rel.split == split].set_index("pert_id")
        ceil = rsub["r_ceiling"].to_dict()
        for m in MODELS:
            path = C.run_path(m, split)
            if not path.exists():
                continue
            pear = per_pert_pearson(pd.read_parquet(path), truth)
            fracs = [pear[p] / ceil[p] for p in pear
                     if p in ceil and np.isfinite(pear[p]) and np.isfinite(ceil[p]) and ceil[p] > 0.05]
            raw = [pear[p] for p in pear if np.isfinite(pear[p])]
            fmed, flo, fhi = cluster_bootstrap_median(fracs)
            rows.append({
                "model": m, "split": split,
                "raw_pearson_median": float(np.nanmedian(raw)),
                "frac_of_ceiling_median": fmed, "frac_ci_lo": flo, "frac_ci_hi": fhi,
                "n_perts": len(fracs),
            })
            print(f"[frac] {m:12s} {split:10s} raw={np.nanmedian(raw):.3f} "
                  f"frac_of_ceiling={fmed:.3f} [{flo:.3f},{fhi:.3f}] n={len(fracs)}", flush=True)
    return pd.DataFrame(rows)


def cross_donor_permutation_null(hvg, n_perm=1000, seed=SEED):
    """Per perturbation, donor-mean SPECIFIC residual (delta - shared); test whether the same
    perturbation's residual reproduces across donors more than random perturbation pairings."""
    rng = np.random.default_rng(seed)
    rows = []
    for split, conds in [(C.SPLIT_CONDITION, (C.CONDITION_HOLDOUT,)), (C.SPLIT_GENE, C.TRAIN_CONDITIONS)]:
        truth = ev.ground_truth(split)
        perts = [p for p in ev.evaluable_perts(split) if p in truth.index]
        deltas = load_czi_deltas(hvg, set(perts), conds)
        shared = truth.loc[perts].to_numpy(float).mean(0)

        # donor-mean specific residual per (pert, donor); keep perts with >=2 donors
        donors = sorted({dn for c in conds for p in perts for (dn, _v) in deltas.get((p, c), [])})
        prof = {}   # pert -> {donor -> residual vec}
        for p in perts:
            byd = {}
            for c in conds:
                for dn, v in deltas.get((p, c), []):
                    byd.setdefault(dn, []).append(v)
            dm = {dn: (np.mean(vs, 0) - shared) for dn, vs in byd.items()}
            if len(dm) >= 2:
                prof[p] = dm
        kept = list(prof)

        def cross_r(mapping):
            # mapping: for each donor pair, list of (residA, residB); corr each, mean
            rs = []
            for i, a in enumerate(donors):
                for b in donors[i + 1:]:
                    for p in kept:
                        pa, pb = mapping(p, a), mapping(p, b)
                        if pa is not None and pb is not None:
                            rs.append(np.corrcoef(pa, pb)[0, 1])
            return float(np.nanmean(rs)) if rs else np.nan

        real = cross_r(lambda p, d: prof[p].get(d))
        # null: within each donor, shuffle which perturbation's residual sits under each label
        null = []
        for _ in range(n_perm):
            shuf = {d: dict(zip(kept, rng.permutation(kept))) for d in donors}  # per-donor pert relabeling
            null.append(cross_r(lambda p, d, s=shuf: prof[s[d][p]].get(d)))
        null = np.array([x for x in null if np.isfinite(x)])
        p_emp = float((null >= real).mean()) if len(null) else np.nan
        rows.append({
            "split": split, "n_perts": len(kept), "n_donors": len(donors),
            "cross_donor_specific_r": real, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "perm_p": p_emp,
        })
        print(f"[xdonor] {split:10s} real_specific_r={real:.4f} null_mean={null.mean():.4f} "
              f"null_p95={np.percentile(null,95):.4f} perm_p={p_emp:.4f} (n={len(kept)})", flush=True)
    return pd.DataFrame(rows)


def main():
    C.ensure_dirs()
    hvg = split_mod.load_hvg()
    rel = pd.read_csv(REL)

    frac = fraction_of_ceiling(rel)
    frac.to_csv(C.RESULTS_DIR / "fraction_of_ceiling.csv", index=False)
    xd = cross_donor_permutation_null(hvg)
    xd.to_csv(C.RESULTS_DIR / "budget_cross_donor.csv", index=False)
    print("\n=== Stage 2 wrote fraction_of_ceiling.csv + budget_cross_donor.csv ===")


if __name__ == "__main__":
    main()
