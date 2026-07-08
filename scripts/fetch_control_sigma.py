#!/usr/bin/env python
"""EXACT C-NL Sigma_control on real CD4 — the best-quality version of the Task-2 residual sizer.

The JEPA .npy cache mixes all guides (unlabeled), so the PROXY sizer (--cell-cache) can only
approximate Sigma_control. This script computes the GENUINE control-cell covariance per
(donor, condition): it downloads each 110-161 GiB assigned_guide single-cell h5ad from the public
S3 mirror, extracts the non-targeting (control) cells, accumulates their log1p(CP10k) HVG covariance,
scores the CIPHER first-order residual for that stratum, then DELETES the raw file and moves on.

Robust + resumable: each stratum's Sigma_c is checkpointed to <sigma-dir>/sigma_<donor>_<cond>.npz,
and per-stratum residual rows are appended to results/cnl_realdata_residual_exact.csv as they finish,
so an interruption (or a stopped box) loses at most the in-flight stratum. Re-run to resume.

Box-run, CPU-only, NO GPU. Needs the aws CLI (public bucket, --no-sign-request) and ~170 GiB free
disk (one file at a time). Reuses the merged sizer (size_cnl_residual) for the math + pseudobulk deltas.

Usage (on the box, after the current build):
  python scripts/fetch_control_sigma.py                          # all 12 strata, 500k control cells each
  python scripts/fetch_control_sigma.py --donors D1 --conditions Rest Stim8hr
  python scripts/fetch_control_sigma.py --dry-run
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/ (for size_cnl_residual)

import numpy as np
import pandas as pd

import core.contract as C
from core.split import load_hvg
from core.data import normalize_pseudobulk_counts, _canon_donor, _canon_condition
import size_cnl_residual as S

BUCKET = "genome-scale-tcell-perturb-seq"
PREFIX = "marson2025_data"
GB = 1 << 30
CHUNK = 20_000


def _head_size(key):
    r = subprocess.run(["aws", "--no-sign-request", "s3api", "head-object", "--bucket", BUCKET,
                        "--key", key, "--query", "ContentLength", "--output", "text"],
                       capture_output=True, text=True)
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() not in ("", "None") else None


def _download(key, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["aws", "--no-sign-request", "s3", "cp", "--only-show-errors",
                        f"s3://{BUCKET}/{key}", str(dest)])
    if r.returncode != 0:
        raise RuntimeError(f"download failed for {key}")


def sigma_control_for_file(path, hvg, cap, seed=0):
    """(Sigma_c over HVG in log1p(CP10k), n_control_cells) from a labeled assigned_guide h5ad.
    Control = non-targeting cells; capped random subsample; chunked float64 covariance."""
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    var = S._strip_version(list(a.var_names))
    col_of = {g: j for j, g in enumerate(var)}
    take = [(i, col_of[g]) for i, g in enumerate(hvg) if g in col_of]
    if not take:
        return None, 0
    hvg_idx = np.array([i for i, _ in take]); var_idx = np.array([j for _, j in take])
    ctrl = np.flatnonzero(S._control_mask(a.obs, path))
    if cap and ctrl.size > cap:
        rng = np.random.default_rng(seed)
        ctrl = np.sort(rng.choice(ctrl, cap, replace=False))
    G = len(hvg)
    s1 = np.zeros(G); s2 = np.zeros((G, G)); n = 0
    for i0 in range(0, ctrl.size, CHUNK):
        sub = a[ctrl[i0:i0 + CHUNK]].to_memory()
        X = sub.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X = normalize_pseudobulk_counts(X)                 # -> log1p(CP10k), ONCE
        Xh = np.zeros((X.shape[0], G)); Xh[:, hvg_idx] = X[:, var_idx]
        s1 += Xh.sum(0); s2 += Xh.T @ Xh; n += Xh.shape[0]
    if n < 2:
        return None, n
    mu = s1 / n
    cov = (s2 - n * np.outer(mu, mu)) / (n - 1)
    return 0.5 * (cov + cov.T), n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", nargs="+", default=["D1", "D2", "D3", "D4"])
    ap.add_argument("--conditions", nargs="+", default=["Rest", "Stim8hr", "Stim48hr"])
    ap.add_argument("--cap-cells", type=int, default=500_000, help="control cells/stratum for Sigma_c")
    ap.add_argument("--tmp-dir", default=str(C.RAW_DIR / "_ctrl_sigma_tmp"))
    ap.add_argument("--sigma-dir", default=str(C.RESULTS_DIR / "cnl_sigma_exact"))
    ap.add_argument("--min-headroom-gb", type=float, default=15.0)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    tmp = Path(args.tmp_dir); sigd = Path(args.sigma_dir); sigd.mkdir(parents=True, exist_ok=True)
    hvg = load_hvg()
    strata = [(d, c) for d in args.donors for c in args.conditions]

    if args.dry_run:
        tot = 0
        for d, c in strata:
            sz = _head_size(f"{PREFIX}/{d}_{c}.assigned_guide.h5ad")
            tot += sz or 0
            print(f"  {d} {c}: {'MISSING' if sz is None else f'{sz/GB:.0f} GiB'}")
        print(f"[plan] {len(strata)} strata, ~{tot/GB:.0f} GiB total sequential; free {shutil.disk_usage(Path.home()).free/GB:.0f} GiB")
        return 0

    deltas = S.load_single_pert_deltas(hvg)                # dict[(donor,cond)] -> (perts, dX)
    out_csv = C.RESULTS_DIR / "cnl_realdata_residual_exact.csv"
    sigmas = {}
    for d, c in strata:
        key = f"{PREFIX}/{d}_{c}.assigned_guide.h5ad"
        dkey = (_canon_donor(d), _canon_condition(c))
        ckpt = sigd / f"sigma_{dkey[0]}_{dkey[1]}.npz"
        if ckpt.exists():                                  # resume: reuse checkpoint, skip download
            z = np.load(ckpt); sigmas[dkey] = (z["cov"], int(z["n"]))
            print(f"[{d} {c}] checkpoint found (n={int(z['n'])}) — skipping download", flush=True)
        else:
            sz = _head_size(key)
            if sz is None:
                print(f"[{d} {c}] MISSING on S3 — skipping", flush=True); continue
            free = shutil.disk_usage(tmp.parent if tmp.parent.exists() else Path.home()).free
            if free < sz + args.min_headroom_gb * GB:
                print(f"[{d} {c}] STOP: needs {sz/GB:.0f} GiB, only {free/GB:.0f} free", flush=True); break
            dest = tmp / f"{d}_{c}.assigned_guide.h5ad"
            t0 = time.time()
            print(f"[{d} {c}] downloading {sz/GB:.0f} GiB ...", flush=True)
            _download(key, dest)
            print(f"[{d} {c}] computing Sigma_control (<= {args.cap_cells:,} control cells) ...", flush=True)
            cov, n = sigma_control_for_file(dest, hvg, args.cap_cells)
            if not args.keep_raw:
                dest.unlink(missing_ok=True)
            if cov is None:
                print(f"[{d} {c}] no control cells found — skipped ({time.time()-t0:.0f}s)", flush=True); continue
            np.savez_compressed(ckpt, cov=cov, n=n)
            sigmas[dkey] = (cov, n)
            print(f"[{d} {c}] Sigma_c ready (n={n:,}, {time.time()-t0:.0f}s); raw deleted", flush=True)

        # incremental scoring: score just this stratum and append, so partial results persist
        if dkey in deltas:
            Rk, nsk = S.score({dkey: sigmas[dkey]}, {dkey: deltas[dkey]}, hvg)
            if not Rk.empty:
                Rk.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
                print(f"[{d} {c}] scored {len(Rk)} perts (skipped {nsk} non-HVG) -> appended", flush=True)

    # final aggregate report over all strata gathered this run + any earlier appends
    if not out_csv.exists():
        print("EXACT run produced no rows — check control-cell availability / strata."); return 1
    R = pd.read_csv(out_csv).drop_duplicates(["donor", "condition", "pert"])
    R.to_csv(out_csv, index=False)
    _report_exact(R)
    (C.RESULTS_DIR / "CNL_SIZER_EXACT_DONE").touch()
    return 0


def _report_exact(R):
    def mi(s):
        s = s.dropna()
        return f"median={s.median():.3f} IQR=[{s.quantile(.25):.3f},{s.quantile(.75):.3f}]" if len(s) else "n/a"
    print("\n================ EXACT C-NL residual (real control-cell Sigma_c) ================")
    print(f"strata={R[['donor','condition']].drop_duplicates().shape[0]}  perts_scored={len(R)}")
    print(f"OVERALL residual fraction (full)       : {mi(R.resid_frac)}")
    print(f"OVERALL residual fraction (trans-only) : {mi(R.resid_frac_trans)}  <- decision-relevant")
    nb = min(5, max(2, R.effect.nunique()))
    try:
        R["bin"] = pd.qcut(R.effect, nb, labels=[f"Q{i+1}" for i in range(nb)], duplicates="drop")
    except ValueError:
        R["bin"] = "all"
    tab = R.groupby("bin", observed=True).agg(
        n=("resid_frac", "size"), effect_med=("effect", "median"),
        full_med=("resid_frac", "median"), trans_med=("resid_frac_trans", "median"))
    print("\nby effect size ||dX|| (Q1 smallest ... largest = decision-relevant):")
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nCaveat: Sigma_c = per-cell log1p(CP10k) control covariance; dX = guide-summed pseudobulk")
    print("delta (same per-gene space, different aggregation). Within (donor,condition).")


if __name__ == "__main__":
    raise SystemExit(main())
