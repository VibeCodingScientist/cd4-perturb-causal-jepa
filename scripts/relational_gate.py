#!/usr/bin/env python
"""C-REL — Relational-Object gate G-R.1 (CPU, committed data only). NO GPU build (G14 fenced).

Five completed negatives all scored the POINTWISE per-perturbation delta δ_p (cross-donor
reproducibility ~0.03, verified). This tests a DIFFERENT object: the RELATIONAL structure over
perturbations (similarity S, program loadings L, per-gene rank R), which averages over many genes
so per-cell noise averages out. Question: does the SPECIFIC (shared-program-removed) relational
structure reproduce cross-donor where the pointwise delta does not?

MANDATORY: shared-program removal. In raw space, nearly every δ loads on the shared activation
program, so raw similarity reproduces ~0.9 for a scientifically empty reason (tautology). We remove
it: δ_specific(p,c,d) = δ(p,c,d) − mean_{p'} δ(p',c,d) per (condition,donor), then pool conditions.
Raw-space reproducibility is reported ONLY as the demoted contrast, never as a pass.

Objects (on specific residuals, per donor, common perts across all donors):
  S — perturbation×perturbation cosine similarity; reproducibility = corr of upper-triangles across donors.
  L — SVD gene factors (shared basis from the donor-mean); per-donor per-pert scores; cross-donor corr per factor.
  R — per-gene rank order of perturbations; cross-donor rank corr over perts, averaged over genes.
Baselines: the 0.03 POINTWISE floor (per-pert cross-donor corr over genes, same data) + a label-
permutation null (shuffle pert labels in one donor). PASS: ≥1 specific object ≥0.30 cross-donor,
clearly above the permutation null. Reads committed CZI + budget_decomposition; CP2/budget untouched.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from core import contract as C
from core import split as split_mod
from budget_reliability import load_czi_deltas

OUT = Path(__file__).resolve().parent.parent
RES, FIG = OUT / "results", OUT / "figures"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
SEED = 42
K_FACTORS = 10
POINTWISE_FLOOR = 0.03   # [VERIFIED committed] per-pert cross-donor δ reproducibility


def _upper(m):
    iu = np.triu_indices(m.shape[0], 1)
    return m[iu]


def cosine_sim(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def build_specific(pcd, donors):
    """per (pert,cond,donor) delta -> specific residual (shared removed per c,d), pooled over conditions."""
    by_cd = defaultdict(list)
    for (p, c, d) in pcd:
        by_cd[(c, d)].append(p)
    spec = {}
    for (c, d), perts in by_cd.items():
        M = np.array([pcd[(p, c, d)] for p in perts])
        shared = M.mean(0)
        for i, p in enumerate(perts):
            spec[(p, c, d)] = M[i] - shared
    spec_pd, raw_pd = {}, {}
    for d in donors:
        perts_all = [p for p in set(pp for (pp, cc, dd) in pcd if dd == d)
                     if all((p, c, d) in spec for c in CONDS)]
        for p in perts_all:
            spec_pd[(p, d)] = np.mean([spec[(p, c, d)] for c in CONDS], 0)
            raw_pd[(p, d)] = np.mean([pcd[(p, c, d)] for c in CONDS], 0)
    return spec_pd, raw_pd


def main():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    hvg = list(split_mod.load_hvg())
    dec = pd.read_csv(RES / "budget_decomposition.csv")
    evaluable = set(dec.pert_id.unique())
    deltas = load_czi_deltas(hvg, evaluable, tuple(CONDS))
    acc = defaultdict(lambda: [np.zeros(len(hvg)), 0])
    for (p, c), lst in deltas.items():
        for (d, v) in lst:
            acc[(p, c, d)][0] += v; acc[(p, c, d)][1] += 1
    pcd = {k: (s / n) for k, (s, n) in acc.items() if n > 0}
    donors = sorted({d for (_p, _c, d) in pcd})
    print(f"[data] {len(pcd)} (pert,cond,donor) cells; {len(donors)} donors", flush=True)

    spec_pd, raw_pd = build_specific(pcd, donors)
    common = sorted(set.intersection(*[{p for (p, d) in spec_pd if d == dn} for dn in donors]))
    print(f"[data] {len(common)} perturbations common to all {len(donors)} donors", flush=True)
    Xs = {d: np.array([spec_pd[(p, d)] for p in common]) for d in donors}   # specific, per donor
    Xr = {d: np.array([raw_pd[(p, d)] for p in common]) for d in donors}    # raw contrast

    def xdonor_S(Xdict, perm=False):
        rs = []
        for i, a in enumerate(donors):
            for b in donors[i + 1:]:
                Sa = _upper(cosine_sim(Xdict[a]))
                Xb = Xdict[b]
                if perm:
                    Xb = Xb[rng.permutation(len(Xb))]
                Sb = _upper(cosine_sim(Xb))
                rs.append(np.corrcoef(Sa, Sb)[0, 1])
        return float(np.mean(rs))

    # ---- S (primary) ----
    S_spec = xdonor_S(Xs); S_raw = xdonor_S(Xr)
    S_null = float(np.mean([xdonor_S(Xs, perm=True) for _ in range(30)]))
    print(f"[S] specific xdonor={S_spec:.3f}  raw(contrast)={S_raw:.3f}  perm-null={S_null:.3f}", flush=True)

    # ---- robustness: is the floor TOTAL, or population-dilution? S among top-effect perts ----
    mag = np.mean([np.linalg.norm(Xs[d], axis=1) for d in donors], 0)   # mean |specific| over donors
    top = np.argsort(-mag)[:200]
    Xs_top = {d: Xs[d][top] for d in donors}
    S_spec_top = xdonor_S(Xs_top)
    S_null_top = float(np.mean([xdonor_S(Xs_top, perm=True) for _ in range(30)]))
    print(f"[S-hi] top-200 high-effect perts: specific xdonor={S_spec_top:.3f}  perm-null={S_null_top:.3f}", flush=True)

    # ---- L (SVD loadings; shared basis from donor-mean specific) ----
    Xbar = np.mean([Xs[d] for d in donors], 0)
    Xc = Xbar - Xbar.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:K_FACTORS]                                   # K×HVG gene factors
    scores = {d: (Xs[d] - Xs[d].mean(0)) @ V.T for d in donors}   # P×K per donor
    L_per_factor = []
    for f in range(K_FACTORS):
        rr = [abs(np.corrcoef(scores[a][:, f], scores[b][:, f])[0, 1])
              for i, a in enumerate(donors) for b in donors[i + 1:]]
        L_per_factor.append(float(np.mean(rr)))
    L_spec = float(np.mean(L_per_factor[:3]))            # top-3 factors
    # perm null for L: shuffle perts in one donor's scores
    L_null = float(np.mean([abs(np.corrcoef(scores[donors[0]][:, 0], scores[donors[1]][rng.permutation(len(common)), 0])[0, 1]) for _ in range(30)]))
    print(f"[L] specific top-3 factor xdonor={L_spec:.3f}  (per-factor {[round(x,2) for x in L_per_factor[:5]]})  perm-null≈{L_null:.3f}", flush=True)

    # ---- R (per-gene rank of perts) ----
    from scipy.stats import spearmanr
    def xdonor_R(Xdict):
        rs = []
        for i, a in enumerate(donors):
            for b in donors[i + 1:]:
                # per-gene spearman over perts, mean over a sample of genes
                gi = rng.choice(Xdict[a].shape[1], 500, replace=False)
                rr = [spearmanr(Xdict[a][:, g], Xdict[b][:, g]).statistic for g in gi]
                rs.append(np.nanmean(rr))
        return float(np.nanmean(rs))
    R_spec = xdonor_R(Xs); R_raw = xdonor_R(Xr)
    print(f"[R] specific per-gene rank xdonor={R_spec:.3f}  raw(contrast)={R_raw:.3f}", flush=True)

    # ---- POINTWISE floor (per-pert cross-donor over genes, same specific data) ----
    def pointwise(Xdict):
        rs = []
        for i, a in enumerate(donors):
            for b in donors[i + 1:]:
                rr = [np.corrcoef(Xdict[a][p], Xdict[b][p])[0, 1] for p in range(len(common))]
                rs.append(np.nanmean(rr))
        return float(np.nanmean(rs))
    pw_spec = pointwise(Xs)
    print(f"[floor] pointwise per-pert cross-donor (specific) = {pw_spec:.3f}  (committed ~{POINTWISE_FLOOR})", flush=True)

    # ---- verdict ----
    passes = {"S": S_spec >= 0.30 and S_spec > S_null + 0.05,
              "L": L_spec >= 0.30 and L_spec > L_null + 0.05,
              "R": R_spec >= 0.30}
    gr1 = any(passes.values())
    rows = [
        {"object": "S_similarity", "specific": S_spec, "raw_contrast": S_raw, "perm_null": S_null, "pass": passes["S"]},
        {"object": "L_loadings_top3", "specific": L_spec, "raw_contrast": np.nan, "perm_null": L_null, "pass": passes["L"]},
        {"object": "R_pergene_rank", "specific": R_spec, "raw_contrast": R_raw, "perm_null": np.nan, "pass": passes["R"]},
        {"object": "S_top200_higheffect", "specific": S_spec_top, "raw_contrast": np.nan, "perm_null": S_null_top, "pass": S_spec_top >= 0.30 and S_spec_top > S_null_top + 0.05},
        {"object": "pointwise_floor", "specific": pw_spec, "raw_contrast": np.nan, "perm_null": np.nan, "pass": False},
    ]
    pd.DataFrame(rows).to_csv(RES / "relational_gate.csv", index=False)
    _fig(rows, S_null)
    print("\n########## G-R.1 ROUTING ##########")
    for k, v in passes.items():
        print(f"  {k}: specific-space {'PASS' if v else 'no'}")
    print(f"  G-R.1 VERDICT: {'PASS -> proceed to G-R.2 (biology, load-bearing)' if gr1 else 'FAIL -> even relations are floored; sixth negative; STOP'}")
    print(f"  (raw-space S={S_raw:.2f} is the shared-program tautology, NOT a pass)")


def _fig(rows, S_null):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.specific, 0.35, label="specific-space (the test)", color="#0d8b96")
    ax.bar(x + 0.2, df.raw_contrast, 0.35, label="raw-space (shared-program tautology, NOT a pass)", color="#c9ccd1")
    ax.axhline(0.30, ls="--", c="k", lw=1, label="PASS bar 0.30")
    ax.axhline(POINTWISE_FLOOR, ls=":", c="gray", lw=1, label=f"pointwise floor {POINTWISE_FLOOR}")
    ax.set_xticks(x); ax.set_xticklabels(df.object, fontsize=8, rotation=15)
    ax.set_ylabel("cross-donor reproducibility"); ax.legend(fontsize=8)
    ax.set_title("G-R.1 — specific-space relational reproducibility vs pointwise floor")
    fig.tight_layout(); fig.savefig(FIG / "relational_gate.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {FIG/'relational_gate.png'}")


if __name__ == "__main__":
    main()
