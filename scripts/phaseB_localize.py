#!/usr/bin/env python
"""Phase B1 — localize the C-NL first-order residual r = ΔX − Σu (CPU, committed data only).

Reads the 12 per-stratum checkpoints cnl_ckpt_donor_*_<cond>.npz (Σ = raw-count control covariance,
ΔX = per-perturbation raw mean shift, genes = 3000 HVG) + the committed residual CSV, and answers
five orthogonal questions, each pointing at a different model class:

  Q1 which genes?      gene-residual profile: concentration + CROSS-STRATUM REPRODUCIBILITY
                       (reproducible program => STRUCTURED; diffuse => leans noise = STOP gate)
  Q2 which perts?      cross-donor reproducibility + concentration of per-pert residual
  Q3 which condition?  DECISIVE — per-condition residual, per-donor, effect/confound-controlled;
                       peaked-at-transition (Rest<Stim8hr>Stim48hr) = transient/far-from-eq = RED,
                       monotone-in-activation = state-dependent = GREEN-ish
  Q4 cell-state dep?   cross-condition same-pert interaction (proxy; cell-level needs raw cells)
  Q5 mean vs distr?    residual is on the MEAN; 2nd-moment (Σ-diagonal) proxy. NOT the 3rd moment.

Single-gene CIPHER fit (matches size_cnl_residual_cipher.py): for perturbation of gene k,
u has only entry k nonzero, u_k = (Σ_k·ΔX)/(Σ_k·Σ_k); pred = Σ[:,k]·u_k; r = ΔX − pred.

Writes results/phaseB_localization.csv, figures/phaseB_localization.png; prints a structured
summary + a DIFFUSE/STRUCTURED verdict. Read-only on committed artifacts; CP2/budget untouched.
"""
from __future__ import annotations
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOX_RESULTS = Path("/home/ubuntu/cd4-perturb-causal-jepa/results")
CKPT_GLOB = str(BOX_RESULTS / "cnl_ckpt_donor_*.npz")
RESID_CSV = BOX_RESULTS / "cnl_realdata_residual_cipher.csv"
OUT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
TOPK_GENE = 50


def parse_stratum(fn):
    m = re.search(r"donor_(\d+)_(\w+)\.npz$", fn)
    return f"donor_{m.group(1)}", m.group(2)


def single_gene_residuals(npz):
    """Return per-pert residual vectors for perts whose gene is a readout gene.
    r[i] = ΔX_i − Σ[:,k_i]·u_i ; resid_frac_i = ||r_i|| / ||ΔX_i||."""
    Sigma = npz["Sigma"]                       # (G,G)
    genes = list(npz["genes"])
    perts = list(npz["perts"])
    dX = npz["dX"]                             # (P,G)
    pos = {g: j for j, g in enumerate(genes)}
    diagSS = np.einsum("ij,ij->i", Sigma, Sigma)   # Σ_k·Σ_k per row
    rows, rfrac, rvecs = [], [], []
    for i, p in enumerate(perts):
        k = pos.get(p)
        if k is None:
            continue
        dx = dX[i]
        nx = np.linalg.norm(dx)
        if nx < 1e-9 or diagSS[k] < 1e-12:
            continue
        u_k = float(Sigma[k] @ dx) / float(diagSS[k])
        r = dx - Sigma[:, k] * u_k
        rows.append(p)
        rfrac.append(float(np.linalg.norm(r) / nx))
        rvecs.append(np.abs(r))
    return genes, rows, np.array(rfrac), (np.array(rvecs) if rvecs else np.zeros((0, len(genes))))


def main():
    RES_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    files = sorted(glob.glob(CKPT_GLOB))
    assert len(files) == 12, f"expected 12 strata, got {len(files)}"
    csv = pd.read_csv(RESID_CSV)

    # --- reconstruct per-stratum gene-residual profiles + per-pert resid_frac ---
    gene_profiles = {}     # (donor,cond) -> (genes, |r| summed over perts normalized)
    pert_rfrac = {}        # (donor,cond) -> dict pert->resid_frac (recomputed)
    genes_ref = None
    selfcheck = []
    for fn in files:
        donor, cond = parse_stratum(fn)
        npz = np.load(fn, allow_pickle=True)
        genes, perts, rfrac, rvecs = single_gene_residuals(npz)
        genes_ref = genes if genes_ref is None else genes_ref
        prof = rvecs.sum(0)
        prof = prof / (prof.sum() + 1e-12)
        gene_profiles[(donor, cond)] = prof
        pert_rfrac[(donor, cond)] = dict(zip(perts, rfrac))
        # self-check vs committed CSV
        sub = csv[(csv.donor == donor) & (csv.condition == cond)].set_index("pert")["resid_frac"]
        common = [p for p in perts if p in sub.index]
        if common:
            mine = np.array([dict(zip(perts, rfrac))[p] for p in common])
            theirs = sub.loc[common].to_numpy()
            selfcheck.append((donor, cond, len(common), float(np.corrcoef(mine, theirs)[0, 1]),
                              float(np.median(np.abs(mine - theirs)))))
    print("=== self-check: recomputed vs committed resid_frac ===")
    for d, c, n, r, mad in selfcheck:
        print(f"  {d} {c:8s} n={n:5d} corr={r:.3f} MAD={mad:.4f}")
    sc_corr = np.median([x[3] for x in selfcheck])
    print(f"  median corr={sc_corr:.3f}  (>0.9 => reconstruction valid)")

    out_rows = []

    # ================= Q1 — which genes? (STRUCTURED vs DIFFUSE = STOP gate) =================
    print("\n=== Q1 — gene-residual profile: concentration + cross-stratum reproducibility ===")
    keys = list(gene_profiles)
    P = np.stack([gene_profiles[k] for k in keys])   # (12, G)
    # concentration: fraction of total residual carried by top-50 genes (of 3000)
    topfrac = np.mean([np.sort(p)[::-1][:TOPK_GENE].sum() for p in P])
    # cross-stratum reproducibility of WHICH genes carry residual
    C = np.corrcoef(P)
    iu = np.triu_indices(len(keys), 1)
    # within-condition (same cond, diff donor) vs cross-condition
    within, cross = [], []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            same_cond = keys[a][1] == keys[b][1]
            (within if same_cond else cross).append(C[a, b])
    print(f"  top-{TOPK_GENE}/3000 genes carry {topfrac*100:.1f}% of residual mass (concentration)")
    print(f"  cross-stratum profile corr: within-condition={np.mean(within):.3f}  cross-condition={np.mean(cross):.3f}  overall={np.mean(C[iu]):.3f}")
    structured = np.mean(C[iu]) > 0.3 and topfrac > 3 * (TOPK_GENE / 3000)
    verdict_q1 = "STRUCTURED" if structured else "DIFFUSE(leans-noise)"
    print(f"  Q1 verdict: {verdict_q1}")
    # top reproducible residual genes (mean profile)
    meanprof = P.mean(0)
    top_idx = np.argsort(-meanprof)[:20]
    top_genes = [(genes_ref[i], float(meanprof[i])) for i in top_idx]
    print("  top-20 residual genes (ENSG):", ",".join(g for g, _ in top_genes[:10]), "...")
    out_rows += [
        {"question": "Q1", "metric": "top50_of_3000_residual_massfrac", "scope": "all", "value": topfrac},
        {"question": "Q1", "metric": "profile_corr_within_condition", "scope": "all", "value": float(np.mean(within))},
        {"question": "Q1", "metric": "profile_corr_cross_condition", "scope": "all", "value": float(np.mean(cross))},
        {"question": "Q1", "metric": "profile_corr_overall", "scope": "all", "value": float(np.mean(C[iu]))},
        {"question": "Q1", "metric": "verdict_structured", "scope": "all", "value": float(structured)},
    ]
    for g, v in top_genes:
        out_rows.append({"question": "Q1", "metric": "top_residual_gene", "scope": g, "value": v})

    # ================= Q2 — which perts? (cross-donor reproducibility) =================
    print("\n=== Q2 — per-pert residual: cross-donor reproducibility + concentration ===")
    for cond in CONDS:
        dons = [d for (d, c) in pert_rfrac if c == cond]
        # pairwise cross-donor corr of per-pert resid_frac
        rr = []
        dl = sorted(set(dons))
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                a, b = pert_rfrac[(dl[i], cond)], pert_rfrac[(dl[j], cond)]
                common = [p for p in a if p in b]
                if len(common) > 20:
                    rr.append(np.corrcoef([a[p] for p in common], [b[p] for p in common])[0, 1])
        cd = float(np.nanmean(rr)) if rr else np.nan
        sub = csv[csv.condition == cond]["resid_frac"]
        cv = float(sub.std() / (sub.mean() + 1e-9))
        print(f"  {cond:8s} cross-donor resid_frac corr={cd:.3f}  CV={cv:.3f}")
        out_rows.append({"question": "Q2", "metric": "cross_donor_pert_corr", "scope": cond, "value": cd})
        out_rows.append({"question": "Q2", "metric": "pert_resid_CV", "scope": cond, "value": cv})

    # ================= Q3 — DECISIVE condition fork =================
    print("\n=== Q3 — DECISIVE: per-condition residual (per-donor, effect/confound-controlled) ===")
    # per donor x condition median (from committed CSV)
    perdon = csv.groupby(["donor", "condition"])["resid_frac"].median().unstack()[CONDS]
    print("  per-donor median resid_frac:")
    print(perdon.round(4).to_string().replace("\n", "\n  "))
    # peaked-at-transition contrast per donor: Stim8hr - mean(Rest,Stim48hr)
    peak = perdon["Stim8hr"] - 0.5 * (perdon["Rest"] + perdon["Stim48hr"])
    print(f"  transition-peak (Stim8hr - mean(Rest,Stim48hr)) per donor: {peak.round(4).to_dict()}")
    print(f"  peak sign consistency: {int((peak>0).sum())}/{len(peak)} donors positive; median={peak.median():.4f}")
    # effect-controlled: OLS resid_frac ~ effect + log n_ctrl + log n_pert + C(condition), donor FE
    d = csv.copy()
    d["leff"] = np.log(d["effect"] + 1e-9); d["lnc"] = np.log(d["n_ctrl"]); d["lnp"] = np.log(d["n_pert"])
    X_cols = ["leff", "lnc", "lnp"]
    Xd = pd.get_dummies(d["condition"], prefix="c")
    Xdon = pd.get_dummies(d["donor"], prefix="d", drop_first=True)
    X = pd.concat([d[X_cols], Xd[["c_Stim8hr", "c_Stim48hr"]], Xdon], axis=1).astype(float)
    X.insert(0, "const", 1.0)
    y = d["resid_frac"].to_numpy()
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    coef = dict(zip(X.columns, beta))
    print(f"  effect-controlled condition effects (ref=Rest): Stim8hr={coef['c_Stim8hr']:+.4f}  Stim48hr={coef['c_Stim48hr']:+.4f}  (beta_effect={coef['leff']:+.4f})")
    monotone = coef["c_Stim48hr"] >= coef["c_Stim8hr"]
    q3_fires = (peak.median() > 0) and ((peak > 0).sum() >= 3) and (coef["c_Stim8hr"] > coef["c_Stim48hr"])
    print(f"  shape: {'MONOTONE in activation (state-dep, GREEN-ish)' if monotone else 'PEAKED at transition (transient, RED)'}")
    print(f"  Q3 FIRES (transition-spike/RED): {q3_fires}")
    for c in CONDS:
        for dn in perdon.index:
            out_rows.append({"question": "Q3", "metric": "resid_frac_median", "scope": f"{dn}/{c}", "value": float(perdon.loc[dn, c])})
    out_rows += [
        {"question": "Q3", "metric": "transition_peak_median", "scope": "all", "value": float(peak.median())},
        {"question": "Q3", "metric": "peak_sign_donors_pos", "scope": "all", "value": float((peak > 0).sum())},
        {"question": "Q3", "metric": "effctrl_Stim8hr_vs_Rest", "scope": "all", "value": float(coef["c_Stim8hr"])},
        {"question": "Q3", "metric": "effctrl_Stim48hr_vs_Rest", "scope": "all", "value": float(coef["c_Stim48hr"])},
        {"question": "Q3", "metric": "beta_effect", "scope": "all", "value": float(coef["leff"])},
        {"question": "Q3", "metric": "q3_fires_transient_RED", "scope": "all", "value": float(q3_fires)},
    ]

    # ================= Q4 — cell-state dependence (proxy) =================
    print("\n=== Q4 — same-pert residual across conditions (state-dependence proxy) ===")
    piv = csv.pivot_table(index=["donor", "pert"], columns="condition", values="resid_frac").dropna(subset=CONDS)
    within_pert_sd = piv[CONDS].std(axis=1)
    print(f"  perts in all 3 conds: {len(piv)}  median within-pert SD across conditions={within_pert_sd.median():.4f}")
    print(f"  (interpretation: large SD => a pert's residual is STATE-dependent; but see Q3 for shape)")
    print("  NOTE: within-condition cell-level activation-coordinate test requires raw cells (not in checkpoints) [needs-cells]")
    out_rows.append({"question": "Q4", "metric": "within_pert_cross_cond_SD_median", "scope": "all", "value": float(within_pert_sd.median())})

    # ================= Q5 — mean vs distribution (2nd-moment proxy, NOT 3rd) =================
    print("\n=== Q5 — mean vs distribution: does residual concentrate in high-variance genes? ===")
    # proxy: correlate mean gene-residual profile with control variance (Σ diagonal), per stratum
    q5 = []
    for fn in files:
        donor, cond = parse_stratum(fn)
        npz = np.load(fn, allow_pickle=True)
        var = np.diag(npz["Sigma"])
        prof = gene_profiles[(donor, cond)]
        m = var > 0
        q5.append(np.corrcoef(np.log(var[m] + 1), prof[m])[0, 1])
    print(f"  median corr(residual-gene-profile, log control-variance)={np.nanmedian(q5):.3f}")
    print("  NOTE: full 2nd-moment (perturbed-cell covariance) recovery needs raw cells [needs-cells]; 3rd moment already orthogonal 12/12 — not re-tested")
    out_rows.append({"question": "Q5", "metric": "corr_residprofile_vs_controlvar", "scope": "all", "value": float(np.nanmedian(q5))})

    df = pd.DataFrame(out_rows)
    df.to_csv(RES_DIR / "phaseB_localization.csv", index=False)
    print(f"\n=== wrote {RES_DIR/'phaseB_localization.csv'} ({len(df)} rows) ===")

    # ---- figure ----
    _figure(perdon, peak, P, C, keys, pert_rfrac, q5, verdict_q1, q3_fires)

    # ---- headline ----
    print("\n########## B1 HEADLINE ##########")
    print(f"Reconstruction self-check corr = {sc_corr:.3f}")
    print(f"Q1 (structured vs diffuse): {verdict_q1}  [STOP gate: {'PROCEED' if structured else 'STOP-diffuse'}]")
    print(f"Q3 (decisive): {'PEAKED@Stim8hr = TRANSIENT/RED' if q3_fires else 'not transition-peaked'}")
    print(f"Route lean: {'RED (measure-and-stop; needs more timepoints)' if q3_fires else 'see Q4/GREEN or ABLATION'}")


def _figure(perdon, peak, P, Cmat, keys, pert_rfrac, q5, verdict_q1, q3_fires):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # panel 1: per-donor per-condition residual (Q3 decisive)
    x = np.arange(3)
    for dn in perdon.index:
        ax[0].plot(x, perdon.loc[dn, CONDS], marker="o", label=dn, alpha=0.8)
    ax[0].plot(x, perdon[CONDS].median(), color="k", lw=3, marker="s", label="median")
    ax[0].set_xticks(x); ax[0].set_xticklabels(CONDS)
    ax[0].set_ylabel("median resid_frac (||ΔX−Σu||/||ΔX||)")
    ax[0].set_title(f"Q3 DECISIVE: residual by state\n{'PEAKED@Stim8hr → transient (RED)' if q3_fires else 'not peaked'}", fontsize=10)
    ax[0].legend(fontsize=7); ax[0].axhline(perdon[CONDS].median().min(), ls=":", c="gray", alpha=0.5)
    # panel 2: cross-stratum gene-profile reproducibility heatmap (Q1 structured/diffuse)
    im = ax[1].imshow(Cmat, vmin=0, vmax=1, cmap="viridis")
    ax[1].set_title(f"Q1: gene-residual profile corr\nacross 12 strata → {verdict_q1}", fontsize=10)
    ax[1].set_xticks(range(len(keys))); ax[1].set_yticks(range(len(keys)))
    ax[1].set_xticklabels([f"{d[-1]}{c[:2]}" for d, c in keys], fontsize=6, rotation=90)
    ax[1].set_yticklabels([f"{d[-1]}{c[:2]}" for d, c in keys], fontsize=6)
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    # panel 3: transition-peak per donor
    ax[2].bar(range(len(peak)), peak.values, color=["#b5179e" if v > 0 else "#888" for v in peak.values])
    ax[2].axhline(0, c="k", lw=0.8)
    ax[2].set_xticks(range(len(peak))); ax[2].set_xticklabels(peak.index, fontsize=8, rotation=45)
    ax[2].set_ylabel("Stim8hr − mean(Rest,Stim48hr)")
    ax[2].set_title("Q3: transition-peak contrast\n(>0 all donors → transient)", fontsize=10)
    fig.suptitle("Phase B1 — localizing the C-NL first-order residual (r = ΔX − Σu)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phaseB_localization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR/'phaseB_localization.png'}")


CONDS = ["Rest", "Stim8hr", "Stim48hr"]
if __name__ == "__main__":
    main()
