"""Task 2 — size the real-data first-order (CIPHER) residual on real CD4.

Measures directly on real data the magnitude the C-NL simulator gate could NOT settle (there the
nonlinear term was ~3-4% of the response). This residual fraction is the go/no-go number for a
real-data C-NL build. Reads only; does NOT start any build.

    residual fraction = || dX_q - Sigma_c u_q || / || dX_q ||
    with dX_q the single-perturbation pseudobulk delta, u_q concentrated on the perturbed gene k,
    Sigma u_q = alpha_q * Sigma_c[:,k],  alpha_q = (Sigma_c[:,k] . dX_q) / ||Sigma_c[:,k]||^2 .

Reported WITHIN (donor, condition) (donor confound applies): overall median + IQR, and STRATIFIED by
effect size ||dX_q|| (low->high). CIPHER's residual grows with magnitude, so the LARGE-EFFECT bin is
the decision-relevant number.

============================================================================================
DATA REALITY  (verified against core/ — read before running; this changed the design)
--------------------------------------------------------------------------------------------
The CP1 path has NO persisted, labeled control SINGLE CELLS:
  * RAW_DIR holds GWCD4i.pseudobulk_merged.h5ad (44.6 GB) — ALREADY PSEUDOBULK (one row per
    guide,donor,condition), not single cells (core/data.py:182-186).
  * The JEPA cache in CELLS_DIR is mmap .npy shards of expression ONLY — no per-cell pert_id
    label, and it already drops Stim48hr/donor_4 (core/models/jepa_data.py:20-28, 307-330).
  * The raw per-guide single-cell h5ads are downloaded then DELETED after JEPA ingest
    (scripts/fetch_jepa_cells.py). Canonical obs pert_id/condition/donor are built in memory at
    build time and never written back — so there is no `contract.harmonize`, and raw obs use
    guide_type / perturbed_gene_id / donor_id / culture_condition (core/data.py:187-206).

So Sigma_c (a control-CELL covariance) needs a genuine cell source. Pick ONE:

  --cells '<glob>'   EXACT. Labeled raw single-cell h5ads (assigned_guide files: obs guide_type /
                     perturbed_gene_id / donor_id / culture_condition, X = raw UMI). Re-fetch them
                     (scripts/fetch_jepa_cells.py --keep-raw) or point at a kept copy. Controls =
                     non-targeting guides; per (donor,condition) covariance in log1p(CP10k) HVG space.
  --cell-cache       PROXY (available now, biased). The JEPA .npy cache: per-(donor,condition) shard
                     covariance of ALL cells (guides mixed) as a stand-in for Sigma_control. Valid
                     only insofar as Sigma_total ~ Sigma_control (NTC-dominant, sparse perturbation);
                     over-estimates baseline covariance, excludes Stim48hr/donor_4. Reported with a
                     loud caveat; use to bracket, not to decide.
  --selftest         Validate the numerical core on synthetic data (no real data, runs anywhere).

BOX: CPU-only, ~64 GB RAM (chunked covariance keeps working RAM modest). NO GPU, NO torch. Run on the
box, report the table, then STOP the box. Underpowered if few perturbed genes are HVG (reported).
============================================================================================
"""

from __future__ import annotations
import argparse
import glob as _glob
import json
import re
import sys
from pathlib import Path

# bootstrap repo root onto sys.path so this runs from scripts/ (repo convention)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import core.contract as C
from core.split import load_hvg
from core.data import normalize_pseudobulk_counts, _canon_condition, _canon_donor

MIN_CTRL_CELLS = 200          # below this a 3000-dim covariance is too rank-deficient to trust
CHUNK = 20_000


def _strip_version(ids):
    """ENSG00000123.4 -> ENSG00000123 (single-cell var ids may carry a version suffix; HVG list
    does not). Reimplemented locally so this script stays torch-free (jepa_data imports torch)."""
    return [str(g).split(".")[0] for g in ids]


# ---------------------------------------------------------------------------
# Streaming covariance (memory-safe; float64 accumulation)
# ---------------------------------------------------------------------------
def _chunked_cov(chunk_iter, n_genes, shrink=0.0):
    n = 0
    s1 = np.zeros(n_genes, dtype=np.float64)
    s2 = np.zeros((n_genes, n_genes), dtype=np.float64)
    for X in chunk_iter:
        X = np.asarray(X, dtype=np.float64)
        n += X.shape[0]
        s1 += X.sum(0)
        s2 += X.T @ X
    if n < 2:
        return None, n
    mu = s1 / n
    cov = (s2 - n * np.outer(mu, mu)) / (n - 1)
    cov = 0.5 * (cov + cov.T)
    if shrink > 0.0:  # shrink toward the diagonal (Ledoit-Wolf-lite) for ill-conditioned Sigma
        d = np.diag(np.diag(cov))
        cov = (1.0 - shrink) * cov + shrink * d
    return cov, n


# ---------------------------------------------------------------------------
# Sigma_c source 1: labeled single-cell h5ads (EXACT)
# ---------------------------------------------------------------------------
def _canon_strata_and_control(obs, path):
    """Return (is_control mask, donor array, condition array) for raw single-cell obs.
    Controls via guide_type ~ non+target (CZI convention); donor/condition from obs if present else
    parsed from the filename (assigned_guide files encode donor_N / condition in the name)."""
    obs = obs.copy()
    cols = {c.lower(): c for c in obs.columns}
    # control mask
    if "guide_type" in cols:
        gt = obs[cols["guide_type"]].astype(str).str.lower()
        is_ctrl = (gt.str.contains("non") & gt.str.contains("target")).to_numpy()
    elif "pert_id" in cols:
        is_ctrl = (obs[cols["pert_id"]].astype(str) == C.CONTROL_PERT_ID).to_numpy()
    else:
        raise KeyError(f"{path}: obs has neither 'guide_type' nor 'pert_id' to identify controls")
    # donor / condition
    fname = Path(path).name
    dcol = next((cols[c] for c in ("donor", "donor_id", "individual") if c in cols), None)
    ccol = next((cols[c] for c in ("condition", "culture_condition", "activation") if c in cols), None)
    donor = (np.array([_canon_donor(v) for v in obs[dcol]]) if dcol
             else _from_filename(fname, r"(donor[_-]?\d+)", "donor"))
    cond = (np.array([_canon_condition(v) for v in obs[ccol]]) if ccol
            else _from_filename(fname, r"(Rest|Stim8hr|Stim48hr)", "condition"))
    if np.ndim(donor) == 0:
        donor = np.full(len(obs), str(donor))
    if np.ndim(cond) == 0:
        cond = np.full(len(obs), str(cond))
    return is_ctrl, donor, cond


def _from_filename(fname, pattern, what):
    m = re.search(pattern, fname, re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot resolve {what} for {fname}: no obs column and no filename token "
                         f"matching /{pattern}/. Rename files as ..._donor_1_Stim8hr... or add obs.")
    return _canon_donor(m.group(1)) if what == "donor" else _canon_condition(m.group(1))


def sigma_from_labeled_cells(paths, hvg, shrink):
    """dict[(donor,cond)] -> (Sigma_c, n_ctrl_cells). Backed reads, control cells only, one log1p."""
    import anndata as ad
    hvg_set = {g: i for i, g in enumerate(hvg)}
    # accumulate per stratum across all files
    acc = {}  # (donor,cond) -> [s1, s2, n]
    for path in paths:
        adata = ad.read_h5ad(path, backed="r")
        var = _strip_version(list(adata.var_names))
        col_of = {g: j for j, g in enumerate(var)}
        take_cols = [(hvg_set[g], col_of[g]) for g in hvg if g in col_of]  # (hvg_idx, var_idx)
        if not take_cols:
            print(f"  WARN {Path(path).name}: no HVG genes in var_names (version mismatch?) — skipped")
            continue
        hvg_idx = np.array([a for a, _ in take_cols]); var_idx = np.array([b for _, b in take_cols])
        is_ctrl, donor, cond = _canon_strata_and_control(adata.obs, path)
        ctrl_rows = np.flatnonzero(is_ctrl)
        for i0 in range(0, ctrl_rows.size, CHUNK):
            rows = ctrl_rows[i0:i0 + CHUNK]
            sub = adata[rows].to_memory()
            X = sub.X
            X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            X = normalize_pseudobulk_counts(X)          # -> log1p(CP10k), ONCE (already includes log1p)
            Xh = np.zeros((X.shape[0], len(hvg)), dtype=np.float64)
            Xh[:, hvg_idx] = X[:, var_idx]
            for key in set(zip(donor[rows], cond[rows])):
                m = (donor[rows] == key[0]) & (cond[rows] == key[1])
                Xk = Xh[m]
                a = acc.setdefault(key, [np.zeros(len(hvg)), np.zeros((len(hvg), len(hvg))), 0])
                a[0] += Xk.sum(0); a[1] += Xk.T @ Xk; a[2] += Xk.shape[0]
    out = {}
    for key, (s1, s2, n) in acc.items():
        if n < MIN_CTRL_CELLS:
            print(f"  WARN stratum {key}: only {n} control cells (<{MIN_CTRL_CELLS}) — skipped")
            continue
        mu = s1 / n
        cov = 0.5 * ((s2 - n * np.outer(mu, mu)) / (n - 1) + ((s2 - n * np.outer(mu, mu)) / (n - 1)).T)
        if shrink > 0:
            cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
        out[key] = (cov, n)
    return out, "labeled control single cells (EXACT)"


# ---------------------------------------------------------------------------
# Sigma_c source 2: JEPA .npy cache (PROXY — all guides, unlabeled)
# ---------------------------------------------------------------------------
def sigma_from_cell_cache(hvg, shrink):
    """dict[(donor,cond)] -> (Sigma_total, n_cells) from the JEPA .npy shards (torch-free reader).
    PROXY: covariance of ALL cells in a (donor,cond) shard (guides mixed) ~ Sigma_control."""
    man_path = C.CELLS_DIR / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(f"no JEPA cache manifest at {man_path}; build it or use --cells.")
    man = json.loads(man_path.read_text())
    if int(man["hvg_n"]) != len(hvg):
        raise ValueError(f"cache HVG panel {man['hvg_n']} != HVG list {len(hvg)} — panels differ.")
    by_key = {}
    for s in man["shards"]:
        if not isinstance(s, dict):
            raise ValueError("cache shards lack (donor,condition) provenance; cannot stratify — use --cells.")
        key = (_canon_donor(str(s.get("donor"))), _canon_condition(str(s.get("condition"))))
        by_key.setdefault(key, []).append(C.CELLS_DIR / s["name"])
    out = {}
    for key, shard_paths in by_key.items():
        def chunks(shard_paths=shard_paths):
            for sp in shard_paths:
                mm = np.load(sp, mmap_mode="r")
                for i in range(0, mm.shape[0], CHUNK):
                    yield np.asarray(mm[i:i + CHUNK])          # already log1p(CP10k) HVG-ordered
        cov, n = _chunked_cov(chunks(), len(hvg), shrink=shrink)
        if cov is not None and n >= MIN_CTRL_CELLS:
            out[key] = (cov, n)
    return out, "JEPA-cache all-cell covariance (PROXY for Sigma_control — biased, excludes Stim48hr/donor_4)"


# ---------------------------------------------------------------------------
# Pseudobulk single-perturbation deltas (confirmed correct against core.contract)
# ---------------------------------------------------------------------------
def load_single_pert_deltas(hvg):
    frames = [pd.read_parquet(p) for p in (C.PSEUDOBULK_TRAIN, C.PSEUDOBULK_TEST) if p.exists()]
    if not frames:
        raise FileNotFoundError(f"no pseudobulk parquet at {C.PSEUDOBULK_DIR}")
    delta = C.pseudobulk_delta(pd.concat(frames)).reindex(columns=hvg)   # index (pert,cond,donor)
    out = {}
    for (pert, cond, donor), row in delta.iterrows():
        if str(pert) == C.CONTROL_PERT_ID:
            continue
        key = (_canon_donor(str(donor)), _canon_condition(str(cond)))
        out.setdefault(key, ([], []))
        out[key][0].append(str(pert)); out[key][1].append(row.to_numpy(dtype=np.float64))
    return {k: (p, np.vstack(v)) for k, (p, v) in out.items()}


# ---------------------------------------------------------------------------
def _resid_frac(col, d):
    denom = float(col @ col); dn = float(np.linalg.norm(d))
    if denom <= 0 or dn <= 0:
        return None
    alpha = float(col @ d) / denom
    return float(np.linalg.norm(d - alpha * col) / dn)


def score(sigmas, deltas, hvg):
    """Per single perturbation: residual fraction of CIPHER's fitted first order.
    resid_frac      = full ||dX - alpha*Sigma[:,k]|| / ||dX||  (task spec).
    resid_frac_trans= same but EXCLUDING the perturbed gene's own coordinate k. The autologous
                      knockdown dominates ||dX||, so the full fraction is biased low; the trans-only
                      number isolates the downstream nonlinearity the C-NL build actually targets."""
    hvg_pos = {g: i for i, g in enumerate(hvg)}
    rows, n_skip = [], 0
    for key, (Sig, n_ctrl) in sigmas.items():
        if key not in deltas:
            continue
        perts, dX = deltas[key]
        for q, pert in enumerate(perts):
            if pert not in hvg_pos:                # target gene not on HVG axis -> can't form Sigma[:,k]
                n_skip += 1
                continue
            k = hvg_pos[pert]
            col, d = Sig[:, k], dX[q]
            rf = _resid_frac(col, d)
            if rf is None:
                continue
            mask = np.ones(len(hvg), dtype=bool); mask[k] = False
            rft = _resid_frac(col[mask], d[mask])
            rows.append(dict(donor=key[0], condition=key[1], pert=pert, n_ctrl=n_ctrl,
                             effect=float(np.linalg.norm(d)), resid_frac=rf,
                             resid_frac_trans=(np.nan if rft is None else rft)))
    return pd.DataFrame(rows), n_skip


def report(R, n_skip, source, out_dir):
    if R.empty:
        print("No scorable perturbations — check the Sigma_c source and that perturbed genes are HVG.")
        return
    print(f"\nSigma_c source: {source}")
    print(f"scored {len(R)} perturbations across {R[['donor','condition']].drop_duplicates().shape[0]} "
          f"(donor,condition) strata; skipped {n_skip} (target gene not in HVG panel).")

    def med_iqr(s):
        s = s.dropna()
        return f"median={s.median():.3f}  IQR=[{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]" if len(s) else "n/a"
    print(f"OVERALL residual fraction (full)        : {med_iqr(R.resid_frac)}")
    print(f"OVERALL residual fraction (trans-only)  : {med_iqr(R.resid_frac_trans)}   "
          f"<- excludes autologous effect; decision-relevant")

    nb = min(5, max(2, R.effect.nunique()))
    try:
        R["effect_bin"] = pd.qcut(R.effect, nb, labels=[f"Q{i+1}" for i in range(nb)], duplicates="drop")
    except ValueError:
        R["effect_bin"] = "all"
    print(f"\nStratified by effect size ||dX|| (Q1=smallest ... largest = decision-relevant):")
    tab = R.groupby("effect_bin", observed=True).agg(
        n=("resid_frac", "size"), effect_median=("effect", "median"),
        resid_full_median=("resid_frac", "median"),
        resid_trans_median=("resid_frac_trans", "median"),
        trans_q25=("resid_frac_trans", lambda s: s.quantile(.25)),
        trans_q75=("resid_frac_trans", lambda s: s.quantile(.75)))
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    R.to_csv(out_dir / "cnl_realdata_residual.csv", index=False)
    tab.to_csv(out_dir / "cnl_realdata_residual_by_effect.csv")
    print(f"\nWrote {out_dir/'cnl_realdata_residual.csv'} (+ _by_effect). Within-donor, within-condition.")
    if "PROXY" in source:
        print("CAVEAT: PROXY covariance (all guides, not control-only). Treat as an upper-ish bracket.")
    print("STOP HERE — do not start the build; stop the box.")


def selftest():
    """Validate the numerical core with no real data: build an SPD Sigma, pick gene k, construct
    dX = alpha*Sigma[:,k] + beta*orthogonal, confirm alpha recovery and residual == beta-fraction."""
    rng = np.random.default_rng(0)
    G, k = 40, 7
    A = rng.standard_normal((G, G)); Sig = A @ A.T / G + np.eye(G)
    col = Sig[:, k]
    orth = rng.standard_normal(G); orth -= (orth @ col) / (col @ col) * col     # perp to col
    orth /= np.linalg.norm(orth)
    ok = True
    for beta in (0.0, 0.25, 1.0):
        dX = 3.0 * col + beta * np.linalg.norm(col) * orth
        alpha = (col @ dX) / (col @ col)
        rf = np.linalg.norm(dX - alpha * col) / np.linalg.norm(dX)
        exp = (beta * np.linalg.norm(col)) / np.linalg.norm(3.0 * col + beta * np.linalg.norm(col) * orth)
        good = abs(alpha - 3.0) < 1e-9 and abs(rf - exp) < 1e-9
        ok &= good
        print(f"  beta={beta:>4}: alpha={alpha:.4f} (want 3.0000)  resid_frac={rf:.4f} (want {exp:.4f})  {'OK' if good else 'FAIL'}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cells", help="glob of labeled single-cell h5ads (EXACT Sigma_control)")
    src.add_argument("--cell-cache", action="store_true", help="JEPA .npy cache (PROXY Sigma)")
    src.add_argument("--selftest", action="store_true", help="numerical self-test, no data")
    ap.add_argument("--shrink", type=float, default=0.0, help="diagonal shrinkage of Sigma_c in [0,1]")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    hvg = load_hvg()
    deltas = load_single_pert_deltas(hvg)
    if args.cells:
        paths = sorted(_glob.glob(args.cells))
        if not paths:
            raise SystemExit(f"--cells matched no files: {args.cells!r}")
        print(f"Sigma_c from {len(paths)} labeled single-cell file(s).")
        sigmas, source = sigma_from_labeled_cells(paths, hvg, args.shrink)
    else:
        sigmas, source = sigma_from_cell_cache(hvg, args.shrink)

    R, n_skip = score(sigmas, deltas, hvg)
    report(R, n_skip, source, C.RESULTS_DIR)


if __name__ == "__main__":
    main()
