#!/usr/bin/env python
"""Phase TC — the gate (CPU, two parts). BOTH must pass to authorize the GPU build.

Idea under test [INFERENCE]: unrecoverable perturbations move the cell ALONG the activation
trajectory (a fixed axis) rather than shifting a fixed endpoint; so the identifiable target is a
SCALAR displacement s_p = proj(delta_p, a) along the activation axis a, not the 3000-dim delta
(proven per-perturbation noise-floored at ~0.03).

G-TC.0 (run first, cheap predictor): does the scalar target reproduce ABOVE the ~0.03 delta floor
AND vary across perturbations? Guarded against the trivial magnitude-coupling explanation (a random
axis + ||delta|| both reproduce just because big effects are big everywhere) by also reporting the
MAGNITUDE-NORMALIZED projection and a random-axis null.

G-TC.1 (geometry claim C-TC.1): does recoverability R_p (do-operator causal_frac_of_ceiling, committed)
fall as trajectory-coupling TC_p = |proj(delta_p,a)|/||delta_p|| rises? PRIMARY bar is the PARTIAL
Spearman(R_p, TC_p | ||delta||, reliability) — otherwise it is just "big/reliable effects are easy."

Activation axis a = Rest->Stim48hr non-targeting CONTROL-state shift (frozen normalization space),
normalized. Robustness: PC-of-condition-contrast + a 2D activation subspace {Stim8hr-Rest, Stim48hr-Rest}.
Reads only committed artifacts (CZI pseudobulk + budget_decomposition.csv + frozen split). CP2/budget
untouched. Writes results/trajectory_coupling_gate.csv + figures/trajectory_coupling_gate.png.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from scipy import stats

from core import contract as C
from core import data as d1data
from core import split as split_mod

CZI = C.RAW_DIR / "GWCD4i.pseudobulk_merged.h5ad"
OUT = Path(__file__).resolve().parent.parent
RES, FIG = OUT / "results", OUT / "figures"
DECOMP = OUT / "results" / "budget_decomposition.csv"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
SEED = 42
DELTA_FLOOR = 0.03            # per-perturbation full-delta cross-donor floor [PR #5/#7 SNR pre-check]


def load_czi(hvg, evaluable):
    """One backed pass. Returns ctrl[(cond,donor)] and guide rows grouped for per-(pert,cond,donor) deltas."""
    import anndata as ad
    a = ad.read_h5ad(CZI, backed="r")
    d1data.ensure_ensembl_var(a)
    genes_all = list(a.var_names)
    gpos = np.array([genes_all.index(g) for g in hvg])
    obs = a.obs
    canon = d1data.czi_obs_to_canonical(obs, d1data.czi_donor_map(obs))
    qmask = d1data._czi_quality_mask(obs)
    cond = canon["condition"].to_numpy(); pert = canon["pert_id"].to_numpy(); donor = canon["donor"].to_numpy()
    want = np.isin(cond, CONDS) & qmask
    is_ctrl = pert == C.CONTROL_PERT_ID
    is_eval = np.isin(pert, list(evaluable))
    rows = np.flatnonzero(want & (is_ctrl | is_eval))
    print(f"[czi] reading {len(rows)} rows on {len(hvg)} HVG ...", flush=True)
    prof = np.empty((len(rows), len(hvg)), np.float32)
    for i0 in range(0, len(rows), 8000):
        sl = np.sort(rows[i0:i0 + 8000])
        X = a[sl].to_memory().X[:, gpos]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        prof[i0:i0 + len(sl)] = d1data.normalize_pseudobulk_counts(X).astype(np.float32)
    rp, rc, rd, ri = pert[rows], cond[rows], donor[rows], is_ctrl[rows]
    donors = sorted(np.unique(rd).tolist())
    ctrl = {}
    for c in CONDS:
        for dn in donors:
            m = (rc == c) & (rd == dn) & ri
            if m.any():
                ctrl[(c, dn)] = prof[m].mean(0)
    # per (pert,cond,donor): mean guide profile -> delta
    from collections import defaultdict
    acc = defaultdict(lambda: [np.zeros(len(hvg)), 0])
    for j in range(len(rows)):
        if ri[j]:
            continue
        k = (rp[j], rc[j], rd[j]); acc[k][0] += prof[j]; acc[k][1] += 1
    delta = {}
    for (p, c, dn), (s, n) in acc.items():
        cb = ctrl.get((c, dn))
        if cb is not None and n > 0:
            delta[(p, c, dn)] = s / n - cb
    return ctrl, delta, donors


def _spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return stats.spearmanr(x[m], y[m]).statistic if m.sum() > 5 else np.nan


def _partial_spearman(y, x, controls):
    """partial Spearman(y, x | controls): rank all, residualize y and x on controls, correlate residuals."""
    def rank(v): return stats.rankdata(v)
    m = np.all(np.isfinite(np.column_stack([y, x] + controls)), axis=1)
    yr, xr = rank(y[m]), rank(x[m])
    Z = np.column_stack([np.ones(m.sum())] + [rank(c[m]) for c in controls])
    by = np.linalg.lstsq(Z, yr, rcond=None)[0]; bx = np.linalg.lstsq(Z, xr, rcond=None)[0]
    ry, rx = yr - Z @ by, xr - Z @ bx
    r, _ = stats.pearsonr(rx, ry)  # pearson on residual ranks = partial spearman
    return r, int(m.sum())


def main():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    hvg = list(split_mod.load_hvg())
    dec = pd.read_csv(DECOMP)
    evaluable = set(dec.pert_id.unique())
    ctrl, delta, donors = load_czi(hvg, evaluable)

    # --- activation axis (per donor + global), + 2D subspace ---
    a_don = {dn: (ctrl[("Stim48hr", dn)] - ctrl[("Rest", dn)]) for dn in donors if ("Stim48hr", dn) in ctrl and ("Rest", dn) in ctrl}
    a_glob = np.mean(list(a_don.values()), 0); a_hat = a_glob / (np.linalg.norm(a_glob) + 1e-12)
    # 2D activation subspace: {Stim8hr-Rest, Stim48hr-Rest} orthonormalized (global)
    b1 = np.mean([ctrl[("Stim8hr", dn)] - ctrl[("Rest", dn)] for dn in donors if ("Stim8hr", dn) in ctrl], 0)
    B = np.column_stack([a_glob, b1]); Q, _ = np.linalg.qr(B)   # Q: HVG x 2 orthonormal
    # robustness axis: PC1 of the control-state means across donors x conditions (dominant axis of
    # variation across states = activation). NOT mean-centered away from the activation signal.
    states = np.array([ctrl[(c, dn)] for c in CONDS for dn in donors if (c, dn) in ctrl])
    _, _, Vt = np.linalg.svd(states - states.mean(0), full_matrices=False)
    a_pc1 = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    if a_pc1 @ a_hat < 0:
        a_pc1 = -a_pc1
    print(f"[axis] cos(control-shift axis, PC1-of-states) = {float(a_hat @ a_pc1):.3f}")
    # random-axis null set
    rand_axes = [rng.standard_normal(len(hvg)) for _ in range(20)]
    rand_axes = [r / np.linalg.norm(r) for r in rand_axes]

    # --- per (pert,cond,donor): scalar projection, magnitude, normalized coupling, subspace coupling ---
    recs = []
    for (p, c, dn), dv in delta.items():
        mag = float(np.linalg.norm(dv))
        if mag < 1e-9:
            continue
        s = float(dv @ a_hat)
        s_sub = float(np.linalg.norm(Q.T @ dv))     # projection magnitude onto 2D activation subspace
        recs.append({"pert": p, "cond": c, "donor": dn, "s": s, "mag": mag,
                     "tc": abs(s) / mag, "tc2d": s_sub / mag, "s_norm": s / mag,
                     "s_pc1": float(dv @ a_pc1), "s_rand": float(dv @ rand_axes[0]) / mag})
    df = pd.DataFrame(recs)

    # ================= G-TC.0 — scalar reproducibility =================
    print("\n=== G-TC.0 — is the scalar target reproducible & non-trivial? ===")
    g0 = []
    for c in CONDS:
        sub = df[df.cond == c]
        piv = sub.pivot_table(index="pert", columns="donor", values="s")
        piv_n = sub.pivot_table(index="pert", columns="donor", values="s_norm")
        piv_m = sub.pivot_table(index="pert", columns="donor", values="mag")
        dd = [c_ for c_ in piv.columns]
        def xdonor(P):
            rr = [_spearman(P[dd[i]].to_numpy(), P[dd[j]].to_numpy())
                  for i in range(len(dd)) for j in range(i + 1, len(dd))]
            return float(np.nanmean(rr))
        # random-axis null: reproducibility of projection onto random axes (magnitude-normalized)
        rnull = []
        for ax in rand_axes[:10]:
            pr = sub.assign(sr=[float(delta[(r.pert, c, r.donor)] @ ax) / r.mag for r in sub.itertuples()]) \
                    .pivot_table(index="pert", columns="donor", values="sr")
            rnull.append(xdonor(pr))
        piv_pc1 = sub.pivot_table(index="pert", columns="donor", values="s_pc1")
        r_s, r_sn, r_mag, r_rand = xdonor(piv), xdonor(piv_n), xdonor(piv_m), float(np.nanmean(rnull))
        r_pc1 = xdonor(piv_pc1)
        var_s = float(sub.groupby("pert").s.mean().std())    # variation of s across perts
        print(f"  {c:8s}: xdonor s={r_s:.3f}  s_norm={r_sn:.3f}  s(PC1)={r_pc1:.3f}  |  mag={r_mag:.3f}  rand-axis-null={r_rand:.3f}  |  SD(s across perts)={var_s:.3f}")
        g0.append({"part": "G-TC.0", "cond": c, "xdonor_s": r_s, "xdonor_s_norm": r_sn, "xdonor_s_pc1": r_pc1,
                   "xdonor_mag": r_mag, "xdonor_randaxis_null": r_rand, "sd_s_across_perts": var_s})
    g0df = pd.DataFrame(g0)
    # decision: scalar carries NON-TRIVIAL signal if normalized projection beats the random-axis null clearly
    sn = g0df.xdonor_s_norm.mean(); null = g0df.xdonor_randaxis_null.mean()
    g0_signal = (sn > 0.15) and (sn > null + 0.10) and (g0df.sd_s_across_perts.mean() > 1e-3)
    print(f"  --> mean s_norm reproducibility={sn:.3f} vs random-axis null={null:.3f}; raw-s reproducibility={g0df.xdonor_s.mean():.3f}")
    print(f"  G-TC.0 VERDICT: {'SIGNAL above floor (non-trivial)' if g0_signal else 'AT/NEAR FLOOR (scalar does not escape the SNR wall)'}")

    # ================= G-TC.1 — recoverability vs trajectory-coupling =================
    print("\n=== G-TC.1 — does recoverability predict trajectory-coupling? (partial is the bar) ===")
    g1 = []
    for split, conds in [(C.SPLIT_CONDITION, ["Stim48hr"]), (C.SPLIT_GENE, ["Rest", "Stim8hr"])]:
        d = df[df.cond.isin(conds)].groupby("pert").agg(tc=("tc", "mean"), tc2d=("tc2d", "mean"), mag=("mag", "mean")).reset_index()
        dsub = dec[dec.split == split][["pert_id", "causal_frac_of_ceiling", "signal_l2", "reliable_r_ceiling"]] \
            .rename(columns={"pert_id": "pert", "causal_frac_of_ceiling": "R", "reliable_r_ceiling": "rel"})
        M = d.merge(dsub, on="pert").dropna(subset=["R", "tc", "mag", "rel"])
        raw = _spearman(M.R.to_numpy(), M.tc.to_numpy())
        par, n = _partial_spearman(M.R.to_numpy(), M.tc.to_numpy(), [M.mag.to_numpy(), M.rel.to_numpy()])
        par2d, _ = _partial_spearman(M.R.to_numpy(), M.tc2d.to_numpy(), [M.mag.to_numpy(), M.rel.to_numpy()])
        # permutation null on the partial correlation
        nulls = []
        for _ in range(2000):
            perm = rng.permutation(M.tc.to_numpy())
            nulls.append(_partial_spearman(M.R.to_numpy(), perm, [M.mag.to_numpy(), M.rel.to_numpy()])[0])
        nulls = np.array(nulls); p_emp = float((np.abs(nulls) >= abs(par)).mean())
        # per-donor sign consistency (per-donor TC vs shared R)
        signs = []
        for dn in donors:
            dpd = df[(df.cond.isin(conds)) & (df.donor == dn)].groupby("pert").agg(tc=("tc", "mean"), mag=("mag", "mean")).reset_index()
            MM = dpd.merge(dsub, on="pert").dropna(subset=["R", "tc", "mag", "rel"])
            if len(MM) > 20:
                pr = _partial_spearman(MM.R.to_numpy(), MM.tc.to_numpy(), [MM.mag.to_numpy(), MM.rel.to_numpy()])[0]
                signs.append(np.sign(pr))
        sign_consistent = int(max((np.array(signs) > 0).sum(), (np.array(signs) < 0).sum())) if signs else 0
        print(f"  {split:9s}: raw Spearman(R,TC)={raw:+.3f}  PARTIAL(|mag,rel)={par:+.3f} (p_perm={p_emp:.4f}, n={n})  "
              f"partial2D={par2d:+.3f}  per-donor sign {sign_consistent}/{len(signs)}")
        g1.append({"part": "G-TC.1", "split": split, "raw_spearman": raw, "partial_spearman": par,
                   "partial_2d": par2d, "perm_p": p_emp, "n": n, "sign_consistent": sign_consistent, "n_donors": len(signs)})
    g1df = pd.DataFrame(g1)
    g1_pass = bool((g1df.partial_spearman.abs() >= 0.3).any() and
                   (g1df.loc[g1df.partial_spearman.abs().idxmax(), "perm_p"] < 0.01) and
                   (g1df.loc[g1df.partial_spearman.abs().idxmax(), "sign_consistent"] >= 3))
    print(f"  G-TC.1 VERDICT (C-TC.1): {'PASS' if g1_pass else 'FAIL'} (bar: |partial|>=0.3, p<0.01, sign>=3/4)")

    # ---- persist ----
    out = pd.concat([g0df.assign(part="G-TC.0"), g1df.assign(part="G-TC.1")], ignore_index=True)
    out.to_csv(RES / "trajectory_coupling_gate.csv", index=False)
    df.to_csv(RES / "trajectory_coupling_perpert.csv", index=False)
    _fig(df, g1df, dec)

    print("\n########## GATE ROUTING ##########")
    print(f"  G-TC.0 (scalar reproduces & non-trivial): {g0_signal}")
    print(f"  G-TC.1 (geometry C-TC.1 passes):          {g1_pass}")
    if g1_pass and g0_signal:
        print("  ROUTE: BUILD (Phase 2) — both gate parts pass.")
    elif g1_pass and not g0_signal:
        print("  ROUTE: report geometry C-TC.1 + scalar-floor finding; DO NOT BUILD.")
    else:
        print("  ROUTE: geometry FAILS -> clean negative; DO NOT BUILD.")


def _fig(df, g1df, dec):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for split, conds in [("condition", ["Stim48hr"]), ("gene", ["Rest", "Stim8hr"])]:
        d = df[df.cond.isin(conds)].groupby("pert").agg(tc=("tc", "mean")).reset_index()
        dsub = dec[dec.split == split][["pert_id", "causal_frac_of_ceiling"]].rename(columns={"pert_id": "pert", "causal_frac_of_ceiling": "R"})
        M = d.merge(dsub, on="pert").dropna()
        a = ax[0] if split == "condition" else ax[1]
        a.scatter(M.tc, M.R, s=6, alpha=0.4)
        a.set_xlabel("trajectory-coupling TC_p = |proj(δ,a)|/‖δ‖"); a.set_ylabel("recoverability R_p (causal frac-of-ceiling)")
        row = g1df[g1df.split == split].iloc[0]
        a.set_title(f"{split}: partial ρ(R,TC|mag,rel)={row.partial_spearman:+.2f} (p={row.perm_p:.3f})")
    fig.suptitle("G-TC.1 — recoverability vs trajectory-coupling")
    fig.tight_layout(); fig.savefig(FIG / "trajectory_coupling_gate.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {FIG/'trajectory_coupling_gate.png'}")


if __name__ == "__main__":
    main()
