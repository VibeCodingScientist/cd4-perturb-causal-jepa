#!/usr/bin/env python
"""G-PA.2 Stage 2b — compute the three cleanly-portable audit probes on Schmidt 2022 (GSE190604),
on the dataset's OWN recomputed floor (never Marson's). Same probe definitions + nulls as the Marson audit;
only the ingestion changes. No do-operator retrain.

Verified in 2a: lanes 1-4 = nostim, 5-8 = stim (activation markers); control = NO-TARGET; 73 perturbed genes,
4 wells/condition. Probes:
  R1 reproducibility floor  — per-perturbation cross-WELL Pearson of delta vectors (wells = replicate units,
     the Marson cross-donor analog), vs a gene-label-permuting null. [Marson cross-donor r = 0.03]
  R2 reliability ceiling    — split-half-over-cells reliability of each perturbation's delta, Spearman-Brown.
  R3 relational-object      — reproducibility of the target×target similarity structure across a well split,
     vs a degree-preserving (row-shuffle) null. [Marson specific S = 0.008]
N/A here: P5 donor (no donor demux in the public form), P4 trajectory (only 2 states), P1/P2 (heavy), P7 (retrain).
"""
import os, gzip, csv, json
from collections import defaultdict
import numpy as np
import scanpy as sc

DEST = os.path.expanduser("~/gpa2-data/raw")
GC = os.path.join(DEST, "GSE190604_cellranger-guidecalls-aggregated-unfiltered.txt.gz")
OUT = os.path.expanduser("~/gpa2/results")
CONTROL = "NO-TARGET"
MIN_CELLS = 15
N_HVG = 2000
B_NULL = 2000
RNG = np.random.default_rng(0)


def load():
    a = sc.read_10x_mtx(DEST, gex_only=True, prefix="GSE190604_")
    a.obs["lane"] = [bc.split("-")[-1] for bc in a.obs_names]
    a.obs["condition"] = np.where(a.obs["lane"].astype(int) <= 4, "nostim", "stim")
    a.obs["well"] = ((a.obs["lane"].astype(int) - 1) % 4 + 1).astype(str)
    # guide singlets -> target
    tgt = {}
    for row in csv.DictReader(gzip.open(GC, "rt"), delimiter="\t"):
        if row.get("num_features", "1") != "1":
            continue
        tgt[row["cell_barcode"]] = row["feature_call"].rsplit("-", 1)[0]
    a.obs["target"] = [tgt.get(bc) for bc in a.obs_names]
    a = a[[t is not None for t in a.obs["target"]]].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat")
    a = a[:, a.var["highly_variable"].to_numpy()].copy()
    print(f"[2b] cells(singlet guide)={a.n_obs} HVG={a.n_vars} targets={a.obs['target'].nunique()}", flush=True)
    return a


def pseudobulk(a):
    """mean HVG vector per (target, condition, well) with >=MIN_CELLS; delta vs NO-TARGET same cond+well."""
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    key = list(zip(a.obs["target"], a.obs["condition"], a.obs["well"]))
    idx = defaultdict(list)
    for i, k in enumerate(key):
        idx[k].append(i)
    means, ncells = {}, {}
    for k, ii in idx.items():
        if len(ii) >= MIN_CELLS:
            means[k] = X[ii].mean(0); ncells[k] = len(ii)
    # delta vs control
    delta = {}
    for (t, c, w), v in means.items():
        if t == CONTROL:
            continue
        ck = (CONTROL, c, w)
        if ck in means:
            delta[(t, c, w)] = v - means[ck]
    return delta, ncells


def _pairwise_r(vecs):
    if len(vecs) < 2:
        return np.nan
    M = np.vstack(vecs)
    C = np.corrcoef(M)
    iu = np.triu_indices(len(vecs), 1)
    return float(np.nanmean(C[iu]))


def r1_reproducibility(delta, cond):
    """cross-well Pearson per target (this condition), + gene-label-permuting null."""
    bytarget = defaultdict(list)
    for (t, c, w), v in delta.items():
        if c == cond:
            bytarget[t].append(v)
    rs = {t: _pairwise_r(vs) for t, vs in bytarget.items() if len(vs) >= 2}
    rs = {t: r for t, r in rs.items() if not np.isnan(r)}
    obs = float(np.median(list(rs.values()))) if rs else np.nan
    # null: permute genes of each well vector independently -> break gene-specific reproducibility
    nulls = []
    tv = {t: vs for t, vs in bytarget.items() if len(vs) >= 2 and t in rs}
    for _ in range(B_NULL // 4):
        vals = []
        for t, vs in tv.items():
            pv = [v[RNG.permutation(len(v))] for v in vs]
            vals.append(_pairwise_r(pv))
        nulls.append(np.nanmedian(vals))
    nulls = np.array(nulls)
    p = (np.sum(nulls >= obs) + 1) / (len(nulls) + 1)
    return obs, float(nulls.mean()), float(nulls.std()), float(p), len(rs)


def r2_reliability(a, cond):
    """split-half-over-cells reliability of each target's delta (this condition), Spearman-Brown."""
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    m = (a.obs["condition"] == cond).to_numpy()
    tg = a.obs["target"].to_numpy()[m]; Xc = X[m]
    ctrl = Xc[tg == CONTROL].mean(0)
    rel = []
    for t in np.unique(tg):
        if t == CONTROL:
            continue
        ii = np.where(tg == t)[0]
        if len(ii) < 2 * MIN_CELLS:
            continue
        ii = RNG.permutation(ii); h = len(ii) // 2
        d1 = Xc[ii[:h]].mean(0) - ctrl; d2 = Xc[ii[h:2 * h]].mean(0) - ctrl
        r = np.corrcoef(d1, d2)[0, 1]
        if not np.isnan(r):
            rel.append(2 * r / (1 + r) if r > -1 else np.nan)   # Spearman-Brown
    rel = [x for x in rel if not np.isnan(x)]
    return float(np.median(rel)) if rel else np.nan, len(rel)


def r3_relational(delta, cond):
    """reproducibility of target×target similarity structure across a well split (specific-space S), vs
    degree-preserving (per-row gene-permute) null. Analog of Marson relational S=0.008."""
    bytarget = defaultdict(dict)
    for (t, c, w), v in delta.items():
        if c == cond:
            bytarget[t][w] = v
    wells = sorted({w for t in bytarget for w in bytarget[t]})
    if len(wells) < 4:
        return np.nan, np.nan, np.nan, 0
    hA, hB = wells[:2], wells[2:]
    targets = [t for t in bytarget if all(w in bytarget[t] for w in wells)]
    if len(targets) < 5:
        return np.nan, np.nan, np.nan, len(targets)
    def mat(hs):
        return np.vstack([np.mean([bytarget[t][w] for w in hs], 0) for t in targets])
    A, Bm = mat(hA), mat(hB)
    def simvec(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        S = Mn @ Mn.T
        return S[np.triu_indices(len(targets), 1)]
    sa, sb = simvec(A), simvec(Bm)
    obs = float(np.corrcoef(sa, sb)[0, 1])
    nulls = []
    for _ in range(B_NULL // 4):
        Ap = np.vstack([r[RNG.permutation(len(r))] for r in A])
        nulls.append(np.corrcoef(simvec(Ap), sb)[0, 1])
    nulls = np.array(nulls)
    p = (np.sum(nulls >= obs) + 1) / (len(nulls) + 1)
    return obs, float(np.nanmean(nulls)), float(p), len(targets)


def main():
    os.makedirs(OUT, exist_ok=True)
    a = load()
    delta, ncells = pseudobulk(a)
    print(f"[2b] pseudobulk delta groups (target,cond,well) with >={MIN_CELLS} cells: {len(delta)}", flush=True)
    rows = []
    for cond in ["nostim", "stim"]:
        o1, n1m, n1s, p1, k1 = r1_reproducibility(delta, cond)
        rel, k2 = r2_reliability(a, cond)
        o3, n3m, p3, k3 = r3_relational(delta, cond)
        print(f"\n==== {cond} ====", flush=True)
        print(f"  R1 reproducibility floor (cross-well r): {o1:.3f}  null {n1m:.3f}±{n1s:.3f} p={p1:.1e}  (n_targets={k1})", flush=True)
        print(f"  R2 reliability ceiling (split-half SB):  {rel:.3f}  (n_targets={k2})", flush=True)
        print(f"  R3 relational-object S (cross-well):     {o3:.3f}  null {n3m:.3f} p={p3:.1e}  (n_targets={k3})", flush=True)
        rows.append(dict(condition=cond, repro_floor_cross_well=round(o1, 4), repro_null=round(n1m, 4),
                         repro_p=p1, n_targets_repro=k1, reliability_ceiling_SB=round(rel, 4),
                         n_targets_rel=k2, relational_S=round(o3, 4), relational_null=round(n3m, 4),
                         relational_p=p3, n_targets_S=k3))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "gpa2_scorecard.csv"), index=False)
    print("\n[2b] wrote results/gpa2_scorecard.csv", flush=True)
    print("MARSON REFERENCE: cross-donor floor 0.03 | relational S 0.008 (both near-floor)", flush=True)
    print("GPA2_STAGE2B_DONE", flush=True)


if __name__ == "__main__":
    main()
