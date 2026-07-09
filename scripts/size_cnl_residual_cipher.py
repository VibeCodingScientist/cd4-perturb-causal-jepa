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
from core.data import _canon_donor, _canon_condition
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


def cipher_stratum(path, gene_mode, min_mean, min_cells=100):
    """One streamed pass over a stratum's assigned_guide h5ad (RAW counts). Schema (verified on the
    real files): obs has guide_type in {targeting, non-targeting, nan}, perturbed_gene_id (ENSG / NTC),
    low_quality, guide_group; var_names are clean ENSG. donor/condition are implicit (per file).
    Returns dict(Sigma, genes, pos, ctrl_mean, deltas={ensembl:(ΔX,n_pert)}, n_ctrl)."""
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    var = S._strip_version(list(a.var_names)); N = a.n_obs
    if gene_mode == "hvg":
        hvg = S._strip_version(load_hvg()); col_of = {g: j for j, g in enumerate(var)}
        cols = np.array([col_of[g] for g in hvg if g in col_of]); genes = [g for g in hvg if g in col_of]
    else:
        cols = None; genes = None

    obs = a.obs
    gt = obs["guide_type"].astype(str).to_numpy()
    pid = np.array(S._strip_version(obs["perturbed_gene_id"].astype(str).to_numpy()))
    lowq = obs["low_quality"].astype(str).to_numpy() if "low_quality" in obs.columns else np.array(["False"] * N)
    ggrp = obs["guide_group"].astype(str).to_numpy() if "guide_group" in obs.columns else np.array([""] * N)
    good = lowq != "True"
    is_ctrl = good & (gt == "non-targeting")
    is_targ = good & (gt == "targeting") & (ggrp != "multi sgRNA")   # single-guide targeting cells

    if gene_mode != "hvg":  # CIPHER gene filter: mean raw count > min_mean over kept cells
        gsum = np.zeros(len(var)); ncnt = 0
        for i0 in range(0, N, CHUNK):
            sl = slice(i0, min(i0 + CHUNK, N)); m = good[sl]
            if m.any():
                X = _densify(a[sl].to_memory().X)[m]; gsum += X.sum(0); ncnt += X.shape[0]
        keep = np.where(gsum / max(ncnt, 1) > min_mean)[0]; cols = keep; genes = [var[j] for j in keep]
    G = len(cols); pos = {g: i for i, g in enumerate(genes)}

    tperts = pd.unique(pid[is_targ]); pidx = {p: i for i, p in enumerate(tperts)}
    PS = np.zeros((len(tperts), G)); PN = np.zeros(len(tperts))
    # perturbed genes that are also readout genes -> compute the third-moment slice T[:,k,k] for each
    pgenes = [p for p in tperts if p in pos]
    pcols = np.array([pos[p] for p in pgenes], dtype=int)
    M = np.zeros((G, len(pcols)))                              # raw 3rd moment sum_c x_i x_k^2 (control)
    s1 = np.zeros(G); s2 = np.zeros((G, G)); nctrl = 0
    for i0 in range(0, N, CHUNK):
        sl = slice(i0, min(i0 + CHUNK, N))
        Xsp = a[sl].to_memory().X
        Xc = _densify(Xsp[:, cols]).astype(np.float64)          # RAW counts over the gene set (no norm)
        ct = is_ctrl[sl]; tg = is_targ[sl]
        if ct.any():
            Xk = Xc[ct]; s1 += Xk.sum(0); s2 += Xk.T @ Xk; nctrl += Xk.shape[0]
            if len(pcols):
                M += Xk.T @ (Xk[:, pcols] ** 2)
        if tg.any():
            ii = np.array([pidx[p] for p in pid[sl][tg]]); np.add.at(PS, ii, Xc[tg]); np.add.at(PN, ii, 1.0)
    if nctrl < 2:
        return None
    ctrl_mean = s1 / nctrl
    Sigma = (s2 - nctrl * np.outer(ctrl_mean, ctrl_mean)) / (nctrl - 1); Sigma = 0.5 * (Sigma + Sigma.T)
    # central third moment slices T_ikk = E[dx_i dx_k^2] from raw moments (control cells)
    Tslices = None
    if len(pcols):
        mu = ctrl_mean; E2 = s2 / nctrl; EM = M / nctrl; d2 = np.diagonal(E2)[pcols]
        Tslices = EM - 2 * mu[pcols][None, :] * E2[:, pcols] - np.outer(mu, d2) + 2 * np.outer(mu, mu[pcols] ** 2)
    deltas = {p: (PS[i] / PN[i] - ctrl_mean, int(PN[i])) for p, i in pidx.items() if PN[i] >= min_cells}
    return dict(Sigma=Sigma, genes=genes, pos=pos, ctrl_mean=ctrl_mean, deltas=deltas, n_ctrl=nctrl,
                Tslices=Tslices, pgenes=pgenes)


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


def cnl_test(res, n_perm=300, seed=0, trans_only=True):
    """Real-data C-NL gate (confound-controlled). Tests whether the baseline third moment T[:,k,k]
    predicts the first-order (CIPHER) residual r = ΔX - u_k* Σ[:,k] BEYOND a null of covariance + mean
    features (the null absorbs the Poisson/mean-driven skew confound). Per-perturbation demeaning
    (=per-pert intercept); incremental ΔR² via Frisch-Waugh; label-permutation null (shuffle which
    perturbation's T-slice aligns to the residual) = the decisive specificity guard.
    Feature folds u_k*^2 so the pooled slope beta is comparable to the theory value 1/2."""
    Sigma, pos, mu = res["Sigma"], res["pos"], res["ctrl_mean"]
    deltas, Tsl, pgenes = res["deltas"], res["Tslices"], res["pgenes"]
    if Tsl is None or not deltas:
        return None
    tcol = {g: j for j, g in enumerate(pgenes)}
    Sig2 = Sigma @ Sigma
    dS = np.diag(Sigma)
    R, BB, fblocks, betas = [], [], [], []
    for p, (dX, npert) in deltas.items():
        if p not in pos or p not in tcol:
            continue
        k = pos[p]; colS = Sigma[:, k]; den = float(colS @ colS)
        if den <= 0:
            continue
        uk = float(colS @ dX) / den
        r = dX - uk * colS
        g = (uk * uk) * Tsl[:, tcol[p]]
        base = np.column_stack([colS, colS ** 2, Sig2[:, k], mu, mu * mu[k], mu * mu[k] ** 2])
        keep = np.ones(len(r), bool)
        if trans_only:
            keep[k] = False                                    # drop autologous (mean-dominated) i=k
        r, g, base = r[keep], g[keep], base[keep]
        r = r - r.mean(); g = g - g.mean(); base = base - base.mean(0)   # per-pert demean
        R.append(r); fblocks.append(g); BB.append(base)
    if len(fblocks) < 5:
        return None
    R = np.concatenate(R); F = np.concatenate(fblocks); B = np.vstack(BB)
    ss_tot = float(R @ R)

    def make_resid(Bm):
        inv = np.linalg.pinv(Bm.T @ Bm)
        return lambda v: v - Bm @ (inv @ (Bm.T @ v))

    resid = make_resid(B)                      # full null: covariance + mean features
    resid_cov = make_resid(B[:, :3])           # covariance-only null (colS, colS^2, Sigma^2)
    rp = resid(R); rp_cov = resid_cov(R)

    def dR2(f, rp_, res_):
        fp = res_(f); sf = float(fp @ fp)
        if sf <= 0 or ss_tot <= 0:
            return 0.0, 0.0
        return float((fp @ rp_) ** 2 / (sf * ss_tot)), float((fp @ rp_) / sf)

    dr2_obs, beta_obs = dR2(F, rp, resid)
    dr2_cov, _ = dR2(F, rp_cov, resid_cov)     # does T beat covariance ALONE (before mean control)?
    raw_corr = float(np.corrcoef(F, R)[0, 1]) if F.std() > 0 and R.std() > 0 else 0.0
    prng = np.random.default_rng(seed); perm = []
    for _ in range(n_perm):
        order = prng.permutation(len(fblocks))
        perm.append(dR2(np.concatenate([fblocks[o] for o in order]), rp, resid)[0])
    perm = np.array(perm)
    return dict(dR2=dr2_obs, dR2_covonly=dr2_cov, beta=beta_obs, raw_corr=raw_corr,
                Fabs=float(np.median(np.abs(F))), Rabs=float(np.median(np.abs(R))),
                perm_p=float((np.sum(perm >= dr2_obs) + 1) / (n_perm + 1)),
                perm_med=float(np.median(perm)), perm_hi=float(np.percentile(perm, 97.5)),
                nperts=len(fblocks), n=len(R))


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
    ap.add_argument("--cnl", action="store_true", help="also run the C-NL third-moment gate per stratum")
    ap.add_argument("--cnl-out", default=str(C.RESULTS_DIR / "cnl_gate_realdata.csv"))
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    C.ensure_dirs(); tmp = Path(args.tmp_dir); out = Path(args.out); cnl_out = Path(args.cnl_out)
    strata = [(d, c) for d in args.donors for c in args.conditions]
    resume_file = cnl_out if args.cnl else out
    done = set()
    if resume_file.exists():
        try:
            dd = pd.read_csv(resume_file, usecols=["donor", "condition"]).drop_duplicates()
            done = set(zip(dd.donor.astype(str), dd.condition.astype(str)))
        except Exception:
            pass
    for d, c in strata:
        if (_canon_donor(d), _canon_condition(c)) in done:
            print(f"[{d} {c}] already in {out.name} — skip (resume)", flush=True); continue
        key = f"{PREFIX}/{d}_{c}.assigned_guide.h5ad"; sz = _head_size(key)
        if sz is None:
            print(f"[{d} {c}] MISSING", flush=True); continue
        free = shutil.disk_usage(tmp.parent if tmp.parent.exists() else Path.home()).free
        if free < sz + args.min_headroom_gb * GB:
            print(f"[{d} {c}] STOP: {sz/GB:.0f} GiB needed, {free/GB:.0f} free", flush=True); break
        dest = tmp / f"{d}_{c}.assigned_guide.h5ad"; t0 = time.time()
        if dest.exists() and abs(dest.stat().st_size - sz) < (1 << 20):
            print(f"[{d} {c}] file already present ({sz/GB:.0f} GiB) — skipping download", flush=True)
        else:
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
        if args.cnl:
            perts = list(res["deltas"])                       # checkpoint for instant re-analysis
            ck = C.RESULTS_DIR / f"cnl_ckpt_{_canon_donor(d)}_{_canon_condition(c)}.npz"
            np.savez_compressed(
                ck, Sigma=res["Sigma"], mu=res["ctrl_mean"],
                Tslices=res["Tslices"] if res["Tslices"] is not None else np.zeros((0, 0)),
                pgenes=np.array(res["pgenes"]), genes=np.array(res["genes"]), perts=np.array(perts),
                dX=np.vstack([res["deltas"][p][0] for p in perts]) if perts else np.zeros((0, len(res["genes"]))))
            for tflag, tag in [(True, "trans"), (False, "full")]:
                ct = cnl_test(res, n_perm=args.n_perm, trans_only=tflag)
                if ct:
                    pd.DataFrame([dict(donor=_canon_donor(d), condition=_canon_condition(c), scope=tag, **ct)]).to_csv(
                        cnl_out, mode="a", header=not cnl_out.exists(), index=False)
                    print(f"[{d} {c}] C-NL {tag}: dR2={ct['dR2']:+.4f} (cov-only {ct['dR2_covonly']:+.4f}) "
                          f"beta={ct['beta']:+.3f} raw_corr={ct['raw_corr']:+.3f} |F|={ct['Fabs']:.2e} |r|={ct['Rabs']:.2e} "
                          f"perm_p={ct['perm_p']:.3f} nperts={ct['nperts']}", flush=True)
    if out.exists():
        R = pd.read_csv(out).drop_duplicates(["donor", "condition", "pert"]); _report(R, f"({args.genes})")
    if args.cnl and cnl_out.exists():
        CN = pd.read_csv(cnl_out).drop_duplicates(["donor", "condition", "scope"])
        print("\n===== C-NL third-moment gate (real data) — does T[:,k,k] beat covariance+mean on the residual =====")
        for tag in ["trans", "full"]:
            sub = CN[CN.scope == tag]
            if sub.empty:
                continue
            v = sub.dR2.to_numpy(); m = float(v.mean())
            if len(v) > 1:                                   # leave-one-stratum-out jackknife SE
                jack = np.array([np.delete(v, i).mean() for i in range(len(v))])
                se = float(np.sqrt((len(v) - 1) / len(v) * np.sum((jack - jack.mean()) ** 2)))
            else:
                se = 0.0
            print(f"  {tag:6s}: strata={len(sub)}  mean ΔR²={m:+.4f}  jackknife 95%CI=[{m-1.96*se:+.4f},{m+1.96*se:+.4f}]"
                  f"  median β={sub.beta.median():+.3f} (theory 0.5)  perm_p median={sub.perm_p.median():.3f}"
                  f"  (perm null ΔR² median={sub.perm_med.median():.4f})")
        CN.to_csv(cnl_out, index=False)
        (C.RESULTS_DIR / "CNL_GATE_DONE").touch()
    if not out.exists() and not (args.cnl and cnl_out.exists()):
        print("no rows produced")


if __name__ == "__main__":
    main()
