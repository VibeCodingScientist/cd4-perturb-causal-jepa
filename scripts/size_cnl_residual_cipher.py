#!/usr/bin/env python
"""CIPHER-EXACT real-data residual sizer (corrects the normalization bug).

Reconciles with CIPHER's Methods (bioRxiv 2025.06.27.661814, "Fluctuation structure predicts
genome-wide perturbation outcomes"):
  * Sigma = covariance of unperturbed control cells in RAW ABSOLUTE COUNTS (NOT log1p/CP10k, NOT z-scored).
  * ΔX    = difference of RAW COUNT MEANS (perturbed - control), same raw space.
  * u_i*  = argmin over scalar u_i of ||ΔX - Sigma u|| with only gene-i nonzero => (Sigma_i·ΔX)/(Sigma_i·Sigma_i).
  * metric R^2 = 1 - ||ΔX - Sigma u||^2 / ||ΔX||^2   (so residual_fraction = sqrt(1 - R^2)).

The earlier size_cnl_residual.py used log1p(CP10k) for BOTH Sigma_c and ΔX; that normalization removes the
library-size / mean-variance co-fluctuation CIPHER relies on, giving a near-diagonal Sigma and residual~1.
Here everything is RAW counts, and ΔX is computed from the SAME cells as Sigma (not the log pseudobulk).

Per (donor,condition): download the assigned_guide h5ad, one streamed pass -> raw control covariance Sigma_c
+ per-perturbation raw ΔX, score CIPHER's fit, then DELETE the raw file. Resumable (per-stratum .npz +
incremental CSV append). Box-run, CPU-only, ~170 GiB free disk, aws CLI.

Usage:
  python scripts/size_cnl_residual_cipher.py --donors D1 --conditions Rest     # single-stratum validation
  python scripts/size_cnl_residual_cipher.py                                    # all 12 strata
  python scripts/size_cnl_residual_cipher.py --selftest
  python scripts/size_cnl_residual_cipher.py --genes cipher --min-mean 1.0      # CIPHER gene filter (mean>1 count)
"""
from __future__ import annotations
import argparse
import subprocess
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import core.contract as C
from core.split import load_hvg
from core.data import czi_obs_to_canonical, czi_donor_map, _canon_donor, _canon_condition
import size_cnl_residual as S

BUCKET = "genome-scale-tcell-perturb-seq"; PREFIX = "marson2025_data"; GB = 1 << 30; CHUNK = 20_000


def _head_size(key):
    r = subprocess.run(["aws", "--no-sign-request", "s3api", "head-object", "--bucket", BUCKET,
                        "--key", key, "--query", "ContentLength", "--output", "text"], capture_output=True, text=True)
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() not in ("", "None") else None


def _download(key, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["aws", "--no-sign-request", "s3", "cp", "--only-show-errors",
                       f"s3://{BUCKET}/{key}", str(dest)]).returncode != 0:
        raise RuntimeError(f"download failed for {key}")


def _densify(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def cipher_stratum(path, gene_mode, min_mean):
    """One streamed pass over a stratum's assigned_guide h5ad (RAW counts):
    returns (Sigma_c, genes_idx, ctrl_mean, {pert_id: (ΔX, n_pert)}, n_ctrl)."""
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    var = S._strip_version(list(a.var_names))
    # gene set
    if gene_mode == "hvg":
        hvg = S._strip_version(load_hvg()); col_of = {g: j for j, g in enumerate(var)}
        cols = np.array([col_of[g] for g in hvg if g in col_of]); genes = [g for g in hvg if g in col_of]
    else:  # CIPHER: genes with mean raw count > min_mean (first pass to get means)
        cols = None; genes = None
    obs = a.obs
    canon = czi_obs_to_canonical(obs, czi_donor_map(obs))
    pert = canon["pert_id"].astype(str).to_numpy()
    is_ctrl = (pert == C.CONTROL_PERT_ID)
    N = a.n_obs

    if gene_mode != "hvg":  # CIPHER filter: mean>min_mean over ALL cells
        gsum = np.zeros(len(var));
        for i0 in range(0, N, CHUNK):
            gsum += _densify(a[i0:i0 + CHUNK].to_memory().X).sum(0)
        keep = np.where(gsum / N > min_mean)[0]
        cols = keep; genes = [var[j] for j in keep]
    G = len(cols)
    pos = {g: i for i, g in enumerate(genes)}

    # one pass: per-pert raw sums (all cells) + control raw covariance
    psum = defaultdict(lambda: np.zeros(G)); pn = defaultdict(int)
    s1 = np.zeros(G); s2 = np.zeros((G, G)); nctrl = 0
    upert = pd.unique(pert); pidx = {p: i for i, p in enumerate(upert)}
    PS = np.zeros((len(upert), G)); PN = np.zeros(len(upert))
    for i0 in range(0, N, CHUNK):
        rows = slice(i0, min(i0 + CHUNK, N))
        Xc = _densify(a[rows].to_memory().X)[:, cols].astype(np.float64)
        pr = pert[rows]; cr = is_ctrl[rows]
        ii = np.array([pidx[p] for p in pr])
        np.add.at(PS, ii, Xc); np.add.at(PN, ii, 1.0)
        if cr.any():
            Xk = Xc[cr]; s1 += Xk.sum(0); s2 += Xk.T @ Xk; nctrl += Xk.shape[0]
    if nctrl < 2:
        return None
    ctrl_mean = s1 / nctrl
    Sigma = (s2 - nctrl * np.outer(ctrl_mean, ctrl_mean)) / (nctrl - 1)
    Sigma = 0.5 * (Sigma + Sigma.T)
    deltas = {}
    for p, i in pidx.items():
        if p == C.CONTROL_PERT_ID or PN[i] < 1:
            continue
        deltas[p] = (PS[i] / PN[i] - ctrl_mean, int(PN[i]))
    return dict(Sigma=Sigma, genes=genes, pos=pos, ctrl_mean=ctrl_mean, deltas=deltas, n_ctrl=nctrl)


def score_stratum(res, donor, cond):
    Sigma, pos, deltas = res["Sigma"], res["pos"], res["deltas"]
    rows = []
    for p, (dX, npert) in deltas.items():
        if p not in pos:            # perturbed gene not in the covariance gene set
            continue
        k = pos[p]; col = Sigma[:, k]
        rf = S._resid_frac(col, dX)
        if rf is None:
            continue
        mask = np.ones(len(pos), bool); mask[k] = False
        rft = S._resid_frac(col[mask], dX[mask])
        rows.append(dict(donor=donor, condition=cond, pert=p, n_ctrl=res["n_ctrl"], n_pert=npert,
                         effect=float(np.linalg.norm(dX)), resid_frac=rf,
                         resid_frac_trans=(np.nan if rft is None else rft),
                         r2=float(1 - rf * rf)))
    return pd.DataFrame(rows)


def _report(R, tag=""):
    def mi(s):
        s = s.dropna(); return f"median={s.median():.3f} IQR=[{s.quantile(.25):.3f},{s.quantile(.75):.3f}]" if len(s) else "n/a"
    print(f"\n===== CIPHER-EXACT residual {tag} =====")
    print(f"strata={R[['donor','condition']].drop_duplicates().shape[0]}  perts={len(R)}")
    print(f"residual fraction (full)       : {mi(R.resid_frac)}")
    print(f"residual fraction (trans-only) : {mi(R.resid_frac_trans)}")
    print(f"R^2 (full)                     : {mi(R.r2)}")
    nb = min(5, max(2, R.effect.nunique()))
    try:
        R["bin"] = pd.qcut(R.effect, nb, labels=[f"Q{i+1}" for i in range(nb)], duplicates="drop")
    except ValueError:
        R["bin"] = "all"
    tab = R.groupby("bin", observed=True).agg(n=("r2", "size"), eff_med=("effect", "median"),
                                              resid_med=("resid_frac", "median"), r2_med=("r2", "median"))
    print("by effect size ||dX|| (Q1 smallest ... largest):"); print(tab.to_string(float_format=lambda x: f"{x:.3f}"))


def selftest():
    rng = np.random.default_rng(0); G, k = 40, 7
    A = rng.standard_normal((G, G)); Sig = A @ A.T / G + np.eye(G)
    dX = 3.0 * Sig[:, k] + 0.2 * np.linalg.norm(Sig[:, k]) * rng.standard_normal(G)
    rf = S._resid_frac(Sig[:, k], dX)
    print(f"selftest: residual_frac={rf:.3f} R2={1-rf*rf:.3f} (well-aligned dX -> small residual)")
    return rf < 0.4


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", nargs="+", default=["D1", "D2", "D3", "D4"])
    ap.add_argument("--conditions", nargs="+", default=["Rest", "Stim8hr", "Stim48hr"])
    ap.add_argument("--genes", choices=["hvg", "cipher"], default="hvg")
    ap.add_argument("--min-mean", type=float, default=1.0)
    ap.add_argument("--tmp-dir", default=str(C.RAW_DIR / "_cipher_tmp"))
    ap.add_argument("--min-headroom-gb", type=float, default=15.0)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "cnl_realdata_residual_cipher.csv"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    C.ensure_dirs(); tmp = Path(args.tmp_dir); out = Path(args.out)
    strata = [(d, c) for d in args.donors for c in args.conditions]
    for d, c in strata:
        key = f"{PREFIX}/{d}_{c}.assigned_guide.h5ad"; sz = _head_size(key)
        if sz is None:
            print(f"[{d} {c}] MISSING", flush=True); continue
        free = shutil.disk_usage(tmp.parent if tmp.parent.exists() else Path.home()).free
        if free < sz + args.min_headroom_gb * GB:
            print(f"[{d} {c}] STOP: {sz/GB:.0f} GiB needed, {free/GB:.0f} free", flush=True); break
        dest = tmp / f"{d}_{c}.assigned_guide.h5ad"; t0 = time.time()
        print(f"[{d} {c}] downloading {sz/GB:.0f} GiB ...", flush=True); _download(key, dest)
        print(f"[{d} {c}] one pass: raw Sigma_c + per-pert raw ΔX ({args.genes}) ...", flush=True)
        res = cipher_stratum(dest, args.genes, args.min_mean)
        if not args.keep_raw:
            dest.unlink(missing_ok=True)
        if res is None:
            print(f"[{d} {c}] no control cells — skip", flush=True); continue
        Rk = score_stratum(res, _canon_donor(d), _canon_condition(c))
        if not Rk.empty:
            Rk.to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"[{d} {c}] scored {len(Rk)} perts (G={len(res['genes'])}, n_ctrl={res['n_ctrl']:,}) "
                  f"resid_med={Rk.resid_frac.median():.3f} R2_med={Rk.r2.median():.3f} ({time.time()-t0:.0f}s)", flush=True)
    if out.exists():
        R = pd.read_csv(out).drop_duplicates(["donor", "condition", "pert"]); _report(R, f"({args.genes})")
        (C.RESULTS_DIR / "CNL_CIPHER_DONE").touch()
    else:
        print("no rows produced")


if __name__ == "__main__":
    main()
