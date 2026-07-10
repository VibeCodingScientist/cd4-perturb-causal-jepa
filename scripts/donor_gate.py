#!/usr/bin/env python
"""C-DON — Donor-Structured Recovery gate (CPU, committed data only). NO GPU build (G13 fenced).

Tests whether the per-perturbation "noise floor" is really a cross-donor-AVERAGING artifact: within
a donor, per-perturbation reproducibility is ~0.48 but cross-donor ~0.03 (a 16x gap noise cannot make).

Step-0 resolved: committed pseudobulk is PER-GUIDE (guide_id = GENE-1/GENE-2, ~2 guides/gene), and
the 2 guides of a gene span both 10x runs (pooled screen) => orthogonal to batch. 10xrun_id kept as a
covariate for the batch check.

Per (guide, donor, condition): effect delta = normalized guide pseudobulk - matched non-targeting
control (same donor+condition), frozen normalization (core.data.normalize_pseudobulk_counts). Work on
the perturbation-SPECIFIC effect s = delta - shared_program(donor,cond) (removes the shared activation
response so the donor main effect / activation composition can't inflate correlations).

G-D.1 (biology vs batch): same-gene independent-guide within-donor concordance vs different-gene, on s
(raw specific) and on s_perp = s - (s.a)a (activation-axis removed = composition control). GO: same >>
diff by >=0.15, p<0.01, >=3/4 donors, surviving composition-correction. Test B: correlate effect with
10xrun / depth / n_cells.

G-D.2 (recoverability): predict an independent held-out guide g2's within-donor specific effect from
(a) donor-AVERAGED cross-donor consensus, (b) donor-CONDITIONED same-donor g1, (c) predict-the-donor-
MEAN. GO: donor-conditioned >= 0.20, above (a) and (c), gain absent under donor-label permutation.

Writes results/donor_structure_gate.csv, figures/donor_structure_gate.png. CP2/budget untouched.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from core import contract as C
from core import data as d1data
from core import split as split_mod

CZI = C.RAW_DIR / "GWCD4i.pseudobulk_merged.h5ad"
OUT = Path(__file__).resolve().parent.parent
RES, FIG = OUT / "results", OUT / "figures"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
SEED = 42
MIN_CELLS = 25


def _corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def load_guide_effects(hvg):
    """Return DataFrame-like dict: per (guide_id, gene, donor, cond) specific effect s over HVG,
    plus donor activation axes and covariates."""
    import anndata as ad
    a = ad.read_h5ad(CZI, backed="r")
    d1data.ensure_ensembl_var(a)
    genes_all = list(a.var_names)
    gpos = np.array([genes_all.index(g) for g in hvg])
    o = a.obs
    gt = o["guide_type"].astype(str).to_numpy()
    keep = o["keep_for_DE"].astype(str).to_numpy() == "True" if "keep_for_DE" in o.columns else np.ones(len(o), bool)
    ncell = o["n_cells"].astype(float).to_numpy()
    donor = o["donor_id"].astype(str).to_numpy()
    cond = o["culture_condition"].astype(str).to_numpy()
    gid = o["guide_id"].astype(str).to_numpy()
    gene = o["perturbed_gene_id"].astype(str).to_numpy()
    run = o["10xrun_id"].astype(str).to_numpy()
    tot = o["total_counts"].astype(float).to_numpy()
    good = keep & (ncell >= MIN_CELLS) & np.isin(cond, CONDS)
    is_ctrl = good & (gt == "non-targeting")
    is_targ = good & (gt == "targeting")
    rows = np.flatnonzero(is_ctrl | is_targ)
    print(f"[czi] reading {len(rows)} good rows on {len(hvg)} HVG ...", flush=True)
    prof = np.empty((len(rows), len(hvg)), np.float32)
    for i0 in range(0, len(rows), 8000):
        sl = np.sort(rows[i0:i0 + 8000])
        X = a[sl].to_memory().X[:, gpos]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        prof[i0:i0 + len(sl)] = d1data.normalize_pseudobulk_counts(X).astype(np.float32)
    rr = {k: v[rows] for k, v in [("gt", gt), ("donor", donor), ("cond", cond), ("gid", gid),
                                  ("gene", gene), ("run", run), ("tot", tot), ("ncell", ncell)]}
    ctrl = {(rr["donor"][j], rr["cond"][j]): None for j in range(len(rows)) if rr["gt"][j] == "non-targeting"}
    for (d, c) in list(ctrl):
        m = (rr["donor"] == d) & (rr["cond"] == c) & (rr["gt"] == "non-targeting")
        ctrl[(d, c)] = prof[m].mean(0)
    donors = sorted({d for (d, _c) in ctrl})
    axis = {}
    for d in donors:
        if (d, "Stim48hr") in ctrl and (d, "Rest") in ctrl:
            v = ctrl[(d, "Stim48hr")] - ctrl[(d, "Rest")]
            axis[d] = v / (np.linalg.norm(v) + 1e-12)
    # per-row delta (targeting only), then specific = delta - shared(donor,cond)
    recs = []
    for j in range(len(rows)):
        if rr["gt"][j] != "targeting":
            continue
        cb = ctrl.get((rr["donor"][j], rr["cond"][j]))
        if cb is None:
            continue
        recs.append((rr["gid"][j], rr["gene"][j], rr["donor"][j], rr["cond"][j], rr["run"][j],
                     rr["tot"][j], rr["ncell"][j], j))
    idx = np.array([r[7] for r in recs])
    delta = prof[idx] - np.array([ctrl[(r[2], r[3])] for r in recs])
    meta = pd.DataFrame([r[:7] for r in recs], columns=["gid", "gene", "donor", "cond", "run", "tot", "ncell"])
    # shared program per (donor,cond); specific s; composition-corrected s_perp
    s = delta.copy()
    for (d, c), g in meta.groupby(["donor", "cond"]).groups.items():
        gi = np.array(list(g))
        s[gi] = delta[gi] - delta[gi].mean(0)
    s_perp = s.copy()
    for d in donors:
        if d not in axis:
            continue
        m = (meta.donor == d).to_numpy()
        proj = s[m] @ axis[d]
        s_perp[m] = s[m] - np.outer(proj, axis[d])
    return meta, s, s_perp, delta, donors


def concordance(meta, mat, donors, rng, n_diff=4000, label=""):
    """same-gene independent-guide vs different-gene concordance, per donor (pooled over conditions)."""
    rows = []
    same_all, diff_all = [], []
    for d in donors:
        md = meta[meta.donor == d]
        # index rows by (gene,cond) -> guide rows
        same, diff = [], []
        by_gene_cond = md.groupby(["gene", "cond"]).indices
        # same-gene: within (gene,cond), pairs of distinct guides
        genes_cond = [k for k, v in by_gene_cond.items() if len(v) >= 2]
        for (g, c) in genes_cond:
            gi = md.index[by_gene_cond[(g, c)]].to_numpy()
            # take first two distinct guides
            gids = md.loc[gi, "gid"].to_numpy()
            uniq = {}
            for r, gd in zip(gi, gids):
                uniq.setdefault(gd, r)
            rs = list(uniq.values())
            if len(rs) >= 2:
                same.append(_corr(mat[rs[0]], mat[rs[1]]))
        # different-gene: random pairs of guides from different genes, same condition
        for c in CONDS:
            mc = md[md.cond == c]
            if len(mc) < 4:
                continue
            arr = mc.index.to_numpy(); genes = mc.gene.to_numpy()
            for _ in range(n_diff // (len(CONDS) * len(donors))):
                i, j = rng.integers(0, len(arr), 2)
                if genes[i] != genes[j]:
                    diff.append(_corr(mat[arr[i]], mat[arr[j]]))
        same = np.array([x for x in same if np.isfinite(x)])
        diff = np.array([x for x in diff if np.isfinite(x)])
        same_all.extend(same.tolist()); diff_all.extend(diff.tolist())
        rows.append({"donor": d, "label": label, "n_same": len(same), "same_med": float(np.median(same)),
                     "diff_med": float(np.median(diff)), "delta": float(np.median(same) - np.median(diff))})
    return pd.DataFrame(rows), np.array(same_all), np.array(diff_all)


def perm_p_from_values(same, diff, rng, n_perm=5000):
    """label-permutation null: shuffle same/diff labels on the pooled concordance values (gene-label
    shuffle == random guide pairing == the different-gene set), compare median gap."""
    same = same[np.isfinite(same)]; diff = diff[np.isfinite(diff)]
    if len(same) < 3 or len(diff) < 3:
        return np.nan
    obs = np.median(same) - np.median(diff)
    pool = np.concatenate([same, diff]); n1 = len(same)
    null = np.array([np.median((p := rng.permutation(pool))[:n1]) - np.median(p[n1:]) for _ in range(n_perm)])
    return float((null >= obs).mean())


def gd2_recovery(meta, s, donors, rng):
    """predict independent held-out guide g2's specific effect: donor-averaged vs donor-conditioned vs donor-mean.
    Precomputed indices (no O(pairs x |guide_vec|) scans)."""
    from collections import defaultdict
    # vectorized per-(gene,donor,gid) mean-over-conditions guide vectors (factorize + add.at)
    key = (meta.gene.astype(str) + "|" + meta.donor.astype(str) + "|" + meta.gid.astype(str)).to_numpy()
    codes, uniq = pd.factorize(key)
    # memory-light fast per-code mean over conditions: sort by code, reduceat (C-level, float32)
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    ss = s[order].astype(np.float32)
    starts = np.concatenate([[0], np.where(np.diff(sc))[0] + 1])
    sums = np.add.reduceat(ss, starts, axis=0)
    cnts = np.diff(np.concatenate([starts, [len(sc)]])).astype(np.float32)
    gvecs = sums / cnts[:, None]            # row i == factorize code i
    print(f"[gd2] built {len(uniq)} guide vectors", flush=True)
    gd_guides = defaultdict(list)          # (gene,donor) -> [guide vectors]
    for i, k in enumerate(uniq):
        g, d, _gid = k.split("|")
        gd_guides[(g, d)].append(gvecs[i])
    gene_donor_mean = defaultdict(dict)    # gene -> donor -> mean guide vec
    donor_acc = {d: [] for d in donors}
    for (g, d), vs in gd_guides.items():
        gene_donor_mean[g][d] = np.mean(vs, 0)
        if d in donor_acc:
            donor_acc[d].extend(vs)
    donor_mean = {d: (np.mean(v, 0) if v else None) for d, v in donor_acc.items()}
    rows = []
    for (g, d), vs in gd_guides.items():
        if len(vs) < 2:
            continue
        g1, g2 = vs[0], vs[1]
        other_ds = [dd for dd in donors if dd != d and dd in gene_donor_mean[g]]
        consensus = np.mean([gene_donor_mean[g][dd] for dd in other_ds], 0) if other_ds else None
        wrong_g1 = gene_donor_mean[g][rng.choice(other_ds)] if other_ds else None
        rows.append({"gene": g, "donor": d,
                     "cond_pred": _corr(g1, g2),                                    # donor-conditioned (same-donor g1)
                     "avg_pred": _corr(consensus, g2) if consensus is not None else np.nan,  # donor-averaged
                     "mean_pred": _corr(donor_mean[d], g2) if donor_mean[d] is not None else np.nan,  # predict-the-donor-mean
                     "perm_pred": _corr(wrong_g1, g2) if wrong_g1 is not None else np.nan})  # donor-permuted
    return pd.DataFrame(rows)


def main():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    hvg = list(split_mod.load_hvg())
    meta, s, s_perp, delta, donors = load_guide_effects(hvg)
    print(f"[data] {len(meta)} targeting guide-rows, {meta.gene.nunique()} genes, {len(donors)} donors")

    # ===== G-D.1 =====
    print("\n=== G-D.1 — guide concordance (biology vs batch) ===")
    c_raw, same_raw, diff_raw = concordance(meta, s, donors, rng, label="specific")
    c_perp, same_perp, diff_perp = concordance(meta, s_perp, donors, rng, label="composition_corrected")
    print("  RAW specific (shared removed):"); print("   " + c_raw.to_string(index=False).replace("\n", "\n   "))
    print("  COMPOSITION-CORRECTED (activation axis removed):"); print("   " + c_perp.to_string(index=False).replace("\n", "\n   "))
    p_raw = perm_p_from_values(same_raw, diff_raw, rng)
    p_perp = perm_p_from_values(same_perp, diff_perp, rng)
    print(f"  perm p (specific)={p_raw:.4f}  perm p (composition-corrected)={p_perp:.4f}")
    # Test B — technical covariate coupling: does effect magnitude/direction track run/depth?
    emag = np.linalg.norm(s, axis=1)
    run_num = pd.factorize(meta.run)[0]
    tb_run = _corr(emag, run_num.astype(float)); tb_depth = _corr(emag, np.log(meta.tot.to_numpy() + 1)); tb_nc = _corr(emag, np.log(meta.ncell.to_numpy() + 1))
    print(f"  Test B covariate coupling: corr(|s|, run)={tb_run:+.3f}  corr(|s|,log depth)={tb_depth:+.3f}  corr(|s|,log ncells)={tb_nc:+.3f}")
    gd1_go = (c_raw.delta.median() >= 0.15 and p_raw < 0.01 and (c_raw.delta > 0).sum() >= 3
              and c_perp.delta.median() >= 0.15 and (c_perp.delta > 0).sum() >= 3)
    print(f"  G-D.1 VERDICT: {'GO (biology, survives composition)' if gd1_go else 'NO-GO (see composition-corrected / delta)'}")

    # ===== G-D.2 =====
    print("\n=== G-D.2 — donor-conditioning recovery (held-out independent guide) ===")
    d2 = gd2_recovery(meta, s, donors, rng)
    med = d2[["cond_pred", "avg_pred", "mean_pred", "perm_pred"]].median()
    print(f"  n pairs={len(d2)}")
    print(f"  donor-CONDITIONED (same-donor g1->g2)   = {med.cond_pred:.3f}")
    print(f"  donor-AVERAGED (cross-donor consensus)  = {med.avg_pred:.3f}")
    print(f"  predict-the-donor-MEAN                  = {med.mean_pred:.3f}")
    print(f"  donor-PERMUTED (wrong-donor g1)         = {med.perm_pred:.3f}")
    gd2_go = (med.cond_pred >= 0.20 and med.cond_pred > med.avg_pred + 0.05
              and med.cond_pred > med.mean_pred + 0.05 and med.cond_pred > med.perm_pred + 0.05)
    print(f"  G-D.2 VERDICT: {'GO (recoverable donor-specific per-perturbation structure)' if gd2_go else 'NO-GO'}")

    # ===== persist + route =====
    out = pd.concat([
        c_raw.assign(test="G-D.1"), c_perp.assign(test="G-D.1"),
        pd.DataFrame([{"test": "G-D.1", "label": "perm_p", "same_med": p_raw, "diff_med": p_perp}]),
        pd.DataFrame([{"test": "G-D.1", "label": "testB_covar", "same_med": tb_run, "diff_med": tb_depth, "delta": tb_nc}]),
        pd.DataFrame([{"test": "G-D.2", "label": "medians", "same_med": med.cond_pred, "diff_med": med.avg_pred,
                       "delta": med.mean_pred, "n_same": len(d2), "donor": f"perm={med.perm_pred:.3f}"}]),
    ], ignore_index=True)
    out.to_csv(RES / "donor_structure_gate.csv", index=False)
    d2.to_csv(RES / "donor_structure_gd2_pairs.csv", index=False)
    _fig(c_raw, c_perp, med)
    print("\n########## GATE ROUTING ##########")
    print(f"  G-D.1 (biology, composition-survived): {gd1_go}")
    print(f"  G-D.2 (recoverable held-out):          {gd2_go}")
    if gd1_go and gd2_go:
        print("  ROUTE: BOTH GO -> build LICENSED (G13). Do NOT execute; lead decides awake.")
    elif gd1_go and not gd2_go:
        print("  ROUTE: real donor biology, NOT per-perturbation-predictable at this depth. Stop.")
    else:
        print("  ROUTE: G-D.1 NO-GO -> floor confirmed / composition / batch-not-excluded. Fifth negative. Stop.")


def _fig(c_raw, c_perp, med):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(c_raw))
    ax[0].bar(x - 0.2, c_raw.same_med, 0.18, label="same-gene (specific)", color="#0d8b96")
    ax[0].bar(x, c_raw.diff_med, 0.18, label="diff-gene (specific)", color="#c9ccd1")
    ax[0].bar(x + 0.2, c_perp.same_med, 0.18, label="same-gene (composition-corr)", color="#b5179e")
    ax[0].set_xticks(x); ax[0].set_xticklabels([d[-4:] for d in c_raw.donor], fontsize=8)
    ax[0].set_ylabel("within-donor guide concordance"); ax[0].legend(fontsize=8)
    ax[0].set_title("G-D.1: same-gene vs different-gene guide concordance")
    ax[1].bar(range(4), [med.cond_pred, med.avg_pred, med.mean_pred, med.perm_pred],
              color=["#0d8b96", "#c9ccd1", "#e2a13b", "#888"])
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["donor-\ncond", "donor-\navg", "donor-\nmean", "donor-\nperm"], fontsize=8)
    ax[1].axhline(0.20, ls="--", c="k", lw=0.8); ax[1].axhline(0.03, ls=":", c="gray", lw=0.8)
    ax[1].set_ylabel("held-out guide prediction (corr)"); ax[1].set_title("G-D.2: recovery vs baselines")
    fig.tight_layout(); fig.savefig(FIG / "donor_structure_gate.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {FIG/'donor_structure_gate.png'}")


if __name__ == "__main__":
    main()
