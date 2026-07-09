#!/usr/bin/env python
"""Phase B — Step 0 SNR pre-check (CPU, committed data only, NO download).

The gate before the 130 GB single-cell run. Phase B found per-perturbation residuals
reproduce cross-donor at ~0.033 (full 3000-gene vector). This asks: is that a
PSEUDOBULK-AGGREGATION artifact that single-cell resolution can fix, or a genuine
EFFECT-SIZE-vs-NOISE floor that more resolution cannot fix?

Key statistical fact: single-cell modeling of a perturbation's MEAN effect does NOT add
cells (pseudobulk is the sufficient statistic for the mean) — the same ~180 cells/pert set
the floor. Single-cell resolution helps ONLY via within-state concentration (effect localized
to a cell subpopulation -> state-conditioning recovers a larger, cleaner effect) or higher
distributional moments. So we:
  (1) calibrate a sampling-noise model against the observed 0.033 (validate the noise model),
  (2) measure reliability + cross-donor reproducibility on each perturbation's own high-effect
      genes and on the cytokine program (the RELEVANT genes, not the null-diluted full vector),
  (3) quantify the levers: how many more CELLS, or how tight a CONCENTRATION fraction f, would
      be needed to lift reproducibility to a usable floor (r~0.30),
  (4) emit ONE go/no-go number + per-condition breakdown (Stim8hr is where the activation
      program inflates variance most — the place single-cell conditioning could help most).

reliability(per-pert, over genes) = 1 - mean_g(v_g/n_p) / var_g(residual_g), where v_g =
control-cell variance (raw counts, = diag Sigma) and n_p = perturbed cells. Expected cross-
donor r ~ reliability. Reads committed .npz + residual CSV only. CP2/budget untouched.
"""
from __future__ import annotations
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

BOX = Path("/home/ubuntu/cd4-perturb-causal-jepa/results")
CKPT_GLOB = str(BOX / "cnl_ckpt_donor_*.npz")
RESID_CSV = BOX / "cnl_realdata_residual_cipher.csv"
OUT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = OUT_DIR / "results"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
TOPK = 50
USABLE = 0.30                       # target per-pert cross-donor reproducibility to call "recoverable"
CYTOKINE = {                        # [VERIFIED] activation/effector program (from B3)
    "ENSG00000111537": "IFNG", "ENSG00000109471": "IL2", "ENSG00000164399": "IL3",
    "ENSG00000169194": "IL13", "ENSG00000164400": "CSF2", "ENSG00000108702": "CCL1",
    "ENSG00000277632": "CCL3", "ENSG00000275302": "CCL4", "ENSG00000169429": "CXCL8",
    "ENSG00000227507": "LTB", "ENSG00000134460": "IL2RA",
}


def parse(fn):
    m = re.search(r"donor_(\d+)_(\w+)\.npz$", fn)
    return f"donor_{m.group(1)}", m.group(2)


def main():
    RES_DIR.mkdir(exist_ok=True)
    csv = pd.read_csv(RESID_CSV)
    npert = {(r.donor, r.condition, r.pert): r.n_pert for r in csv.itertuples()}

    # reconstruct per-pert residual + keep dX, per-gene noise var; collect for cross-donor step
    strat = {}   # (donor,cond) -> dict(genes,pos, pert-> (dx, resid, nvar_g, n_p))
    rows = []
    for fn in sorted(glob.glob(CKPT_GLOB)):
        d, c = parse(fn)
        z = np.load(fn, allow_pickle=True)
        Sigma, genes, perts, dX = z["Sigma"], list(z["genes"]), list(z["perts"]), z["dX"]
        pos = {g: j for j, g in enumerate(genes)}
        vg = np.clip(np.diag(Sigma), 1e-8, None)          # control-cell variance (raw counts)
        diagSS = np.einsum("ij,ij->i", Sigma, Sigma)
        cyto_cols = [pos[g] for g in CYTOKINE if g in pos]
        store = {"genes": genes, "pos": pos, "perts": {}}
        for i, p in enumerate(perts):
            k = pos.get(p)
            n_p = npert.get((d, c, p))
            if k is None or n_p is None or diagSS[k] < 1e-12:
                continue
            dx = dX[i]; nx = np.linalg.norm(dx)
            if nx < 1e-9:
                continue
            u = float(Sigma[k] @ dx) / float(diagSS[k])
            r = dx - Sigma[:, k] * u
            nvar = vg / n_p                                # sampling variance of the mean shift, per gene
            store["perts"][p] = (dx, r, nvar, int(n_p))
            top = np.argsort(-np.abs(dx))[:TOPK]
            rel_full = 1 - np.mean(nvar) / (np.var(r) + 1e-12)
            rel_top = 1 - np.mean(nvar[top]) / (np.var(r[top]) + 1e-12)
            snr_top = float(np.linalg.norm(dx[top]) / np.sqrt(np.sum(nvar[top]) + 1e-12))
            snr_cyto = (float(np.linalg.norm(dx[cyto_cols]) / np.sqrt(np.sum(nvar[cyto_cols]) + 1e-12))
                        if cyto_cols else np.nan)
            rows.append({"donor": d, "condition": c, "pert": p, "n_p": int(n_p),
                         "rel_full": float(np.clip(rel_full, 0, 1)),
                         "rel_top": float(np.clip(rel_top, 0, 1)),
                         "snr_top": snr_top, "snr_cyto": snr_cyto})
        strat[(d, c)] = store
    df = pd.DataFrame(rows)

    # cross-donor reproducibility on each pert's own top-effect genes (the relevant subset)
    def xdonor(top_only):
        donors = sorted({d for (d, _c) in strat})
        per_cond = {}
        for c in CONDS:
            rr = []
            for i, a in enumerate(donors):
                for b in donors[i + 1:]:
                    sa, sb = strat.get((a, c)), strat.get((b, c))
                    if not sa or not sb:
                        continue
                    for p in sa["perts"]:
                        if p in sb["perts"]:
                            dxa, ra, _, _ = sa["perts"][p]
                            _, rb, _, _ = sb["perts"][p]
                            if top_only:
                                idx = np.argsort(-np.abs(dxa))[:TOPK]
                                x, y = ra[idx], rb[idx]
                            else:
                                x, y = ra, rb
                            if x.std() > 1e-9 and y.std() > 1e-9:
                                rr.append(np.corrcoef(x, y)[0, 1])
            per_cond[c] = float(np.nanmean(rr)) if rr else np.nan
        return per_cond

    xd_full = xdonor(False)
    xd_top = xdonor(True)

    print("=== CALIBRATION (validate the sampling-noise model) ===")
    print(f"  observed cross-donor r (full 3000 genes): {np.nanmean(list(xd_full.values())):.3f}  (Phase B reported ~0.033)")
    print(f"  model-predicted reliability (full, median): {df.rel_full.median():.3f}")
    print("  -> if these agree, the ~0.033 full-vector floor IS sampling-noise-consistent.\n")

    print("=== ON THE RELEVANT GENES (per-pert top-50 effect genes) ===")
    print(f"  cross-donor r on top-50 effect genes (median over conds): {np.nanmean(list(xd_top.values())):.3f}")
    for c in CONDS:
        print(f"    {c:8s}: xdonor_full={xd_full[c]:.3f}  xdonor_top50={xd_top[c]:.3f}  "
              f"rel_top(med)={df[df.condition==c].rel_top.median():.3f}  "
              f"snr_top(med)={df[df.condition==c].snr_top.median():.2f}  "
              f"snr_cyto(med)={df[df.condition==c].snr_cyto.median():.2f}")

    print("\n=== EFFECT DETECTABILITY ===")
    det = (df.snr_top > 3).mean()
    det_cyto = (df.snr_cyto > 3).mean()
    print(f"  fraction of perts with top-gene effect SNR>3 (clearly detectable): {det*100:.0f}%")
    print(f"  fraction with cytokine-gene SNR>3: {det_cyto*100:.0f}%")

    print("\n=== LEVERS to lift per-pert reproducibility to a usable floor (r={:.2f}) ===".format(USABLE))
    # current top-gene reliability / reproducibility is the base
    r_now = np.nanmean(list(xd_top.values()))
    # reliability(n) = S/(S+N/n_factor). r_now ~ S/(S+N0). To reach USABLE: N_needed/N0 = (S/N0)*(1-U)/U ... derive cell factor
    # from r_now = S/(S+N0): S/N0 = r_now/(1-r_now). Want U = S/(S+N0/k): => k = (N0/S)*(U/(1-U)) = (1-r_now)/r_now * U/(1-U)
    if r_now > 0:
        cells_factor = ((1 - r_now) / r_now) * (USABLE / (1 - USABLE))
        cells_factor = 1.0 / cells_factor if cells_factor > 0 else np.inf   # k = more-cells multiple
        conc_f = cells_factor ** -1  # concentration fraction f ~ 1/k^? ; report both cleanly below
    else:
        cells_factor = np.inf
    # cleaner: k (cell multiple) solving U = (S/N0*k)/(S/N0*k+1) with s0=S/N0=r_now/(1-r_now)
    s0 = r_now / (1 - r_now) if r_now < 1 else np.inf
    k_cells = (USABLE / (1 - USABLE)) / s0 if s0 > 0 else np.inf     # cell multiple needed
    f_conc = 1.0 / (k_cells ** 2) if np.isfinite(k_cells) and k_cells > 0 else np.nan  # SNR boost 1/sqrt(f)=sqrt(k)->f=1/k^2? see note
    print(f"  current top-gene cross-donor r (base) = {r_now:.3f}  -> implied signal/noise s0 = {s0:.3f}")
    print(f"  MORE-CELLS lever: need ~{k_cells:.1f}x more cells/pert to reach r={USABLE:.2f} (single-cell does NOT add cells)")
    print(f"  CONCENTRATION lever: state-conditioning boosts SNR by 1/sqrt(f); to reach r={USABLE:.2f} the effect")
    print(f"    must be concentrated in f~{max(0.0,min(1.0,1.0/k_cells)):.2f} of cells (SNR boost sqrt(k)={np.sqrt(k_cells):.1f}x)")

    # GO/NO-GO
    print("\n########## GO / NO-GO ##########")
    green = (np.nanmean(list(xd_top.values())) >= 0.10) and (det >= 0.20)
    HEADLINE = np.nanmean(list(xd_top.values()))
    print(f"  HEADLINE NUMBER — per-pert cross-donor reproducibility on relevant (top-50) genes = {HEADLINE:.3f}")
    print(f"    (vs {np.nanmean(list(xd_full.values())):.3f} full-vector; usable target {USABLE:.2f}; {det*100:.0f}% of perts detectable)")
    print(f"  CALL: {'GREEN (recoverable structure on relevant genes; single-cell concentration plausibly helps)' if green else 'NOT GREEN (per-pert frontier ~noise-limited even on relevant genes; single-cell cannot add cells)'}")

    df.to_csv(RES_DIR / "phaseB_snr_precheck.csv", index=False)
    summ = pd.DataFrame([
        {"metric": "xdonor_full_median", "value": float(np.nanmean(list(xd_full.values())))},
        {"metric": "xdonor_top50_median", "value": float(HEADLINE)},
        {"metric": "rel_full_median", "value": float(df.rel_full.median())},
        {"metric": "rel_top_median", "value": float(df.rel_top.median())},
        {"metric": "frac_detectable_snr3", "value": float(det)},
        {"metric": "frac_cyto_detectable_snr3", "value": float(det_cyto)},
        {"metric": "cells_multiple_to_usable", "value": float(k_cells)},
        {"metric": "green", "value": float(green)},
    ])
    summ.to_csv(RES_DIR / "phaseB_snr_precheck_summary.csv", index=False)
    print(f"\nwrote {RES_DIR/'phaseB_snr_precheck.csv'} + _summary.csv")


if __name__ == "__main__":
    main()
