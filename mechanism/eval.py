"""Aggregate C4 records into the deliverables.

Inputs : results/c4_records.csv   (from run_c4.py)
Outputs: results/c4_auroc.csv     (per method x mode + pooled: AUROC & Spearman, median + 95% CI)
         results/moneyshot.png    (predicted transportability vs true continuous agreement)

Metrics
-------
Pooled AUROC  : predicted score vs the binary transportability label, pooled over all
                four modes (both classes present: none/b transportable, a/both blocked).
Per-mode      : Spearman(predicted score, continuous agreement). Only well-defined where
                the continuous agreement varies within the mode (the rewiring modes a/both);
                for none/b the agreement is ~1 for every perturbation (same A) so Spearman is
                reported as NaN.
Uncertainty   : cluster bootstrap over (mode, seed) instances (perturbations within an
                instance are correlated), 2000 resamples, 95% percentile interval. CIs are
                reported for each method and for the mechanism-minus-null gaps.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META = ["mode", "seed", "gene", "label", "agree"]
N_BOOT = 2000
BOOT_SEED = 12345

# Primary methods reported in the money-shot (subset of the score columns present).
PRIMARY = ["mechanism", "corr_null", "gears_null"]
PRETTY = {
    "mechanism": "Mechanism (Â, brief inversion)",
    "mechanism_col": "Mechanism (Â column-k)",
    "mech_col_latent": "Mechanism ceiling (Â from noiseless τ)",
    "true_col": "Oracle (true A columns)",
    "corr_null": "Null: correlation graph",
    "gears_null": "Null: co-expression (GEARS-style)",
}


def _auroc(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def cluster_bootstrap(df, method_cols, n_boot=N_BOOT):
    """Resample (mode, seed) instances with replacement; recompute pooled AUROC per method
    and the mechanism-minus-null gaps. Returns dict of percentile summaries."""
    rng = np.random.default_rng(BOOT_SEED)
    groups = {k: g for k, g in df.groupby(["mode", "seed"])}
    keys = list(groups.keys())
    idx = np.arange(len(keys))

    aurocs = {m: [] for m in method_cols}
    gaps = {}
    has_mech = "mechanism" in method_cols
    null_cols = [c for c in ("corr_null", "gears_null") if c in method_cols]
    if has_mech:
        for nc in null_cols:
            gaps[f"mechanism-{nc}"] = []
        if "mechanism_col" in method_cols:
            for nc in null_cols:
                gaps[f"mechanism_col-{nc}"] = []

    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        sub = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
        lab = sub["label"].to_numpy()
        cur = {}
        for m in method_cols:
            cur[m] = _auroc(lab, sub[m].to_numpy())
            aurocs[m].append(cur[m])
        for g in gaps:
            a, b = g.split("-")
            gaps[g].append(cur.get(a, np.nan) - cur.get(b, np.nan))

    def summ(v):
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        return dict(median=float(np.median(v)), lo=float(np.percentile(v, 2.5)),
                    hi=float(np.percentile(v, 97.5)))

    return {m: summ(a) for m, a in aurocs.items()}, {g: summ(a) for g, a in gaps.items()}


def per_mode_spearman(df, method_cols):
    """Spearman(predicted, continuous agreement) per mode, with a seed-cluster bootstrap CI."""
    rng = np.random.default_rng(BOOT_SEED + 1)
    rows = []
    for mode, dm in df.groupby("mode"):
        agree = dm["agree"].to_numpy()
        agree_varies = np.std(agree) > 1e-9
        seeds = sorted(dm["seed"].unique())
        gs = {s: dm[dm["seed"] == s] for s in seeds}
        for m in method_cols:
            if not agree_varies:
                rows.append(dict(mode=mode, method=m, spearman=np.nan, lo=np.nan, hi=np.nan))
                continue
            rho = spearmanr(dm[m], agree).correlation
            boot = []
            for _ in range(N_BOOT):
                pick = rng.choice(seeds, size=len(seeds), replace=True)
                sub = pd.concat([gs[s] for s in pick], ignore_index=True)
                if np.std(sub["agree"]) < 1e-9:
                    continue
                boot.append(spearmanr(sub[m], sub["agree"]).correlation)
            boot = np.array(boot, float); boot = boot[~np.isnan(boot)]
            lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
            rows.append(dict(mode=mode, method=m, spearman=float(rho), lo=float(lo), hi=float(hi)))
    return pd.DataFrame(rows)


def moneyshot(df, path):
    """Predicted transportability (x) vs true continuous agreement (y); the mechanism (its best
    variant, column-k) colored, nulls grey. A mechanism cloud tight on the diagonal would be the
    'we recover the condition' picture. The mechanism's best variant is plotted so the visual
    gives the claim its most favorable honest representation."""
    lab = df["label"].to_numpy()
    mech_col = "mechanism_col" if "mechanism_col" in df.columns else "mechanism"
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    grey = [c for c in ["corr_null", "gears_null"] if c in df.columns]
    styles = {
        "corr_null": dict(color="#8a9099", marker="s", s=20, alpha=0.40, zorder=1),
        "gears_null": dict(color="#c2c7cd", marker="^", s=20, alpha=0.40, zorder=1),
    }
    for m in grey:
        au = _auroc(lab, df[m].to_numpy())
        ax.scatter(df[m], df["agree"], label=f"{PRETTY.get(m, m)}  (AUROC {au:.2f})", **styles[m])
    au_m = _auroc(lab, df[mech_col].to_numpy())
    ax.scatter(df[mech_col], df["agree"], color="#1f77b4", marker="o", s=30, alpha=0.85, zorder=3,
               edgecolors="white", linewidths=0.3,
               label=f"{PRETTY.get(mech_col, mech_col)}  (AUROC {au_m:.2f})")
    ax.plot([-1, 1], [-1, 1], ls="--", lw=1, color="black", alpha=0.4, zorder=2, label="y = x")
    ax.axhline(0.9, ls=":", lw=1, color="crimson", alpha=0.6, zorder=2)
    ax.text(-1.0, 0.905, "transportable threshold (agreement ≥ 0.9)", color="crimson", fontsize=8, va="bottom")

    # reference AUROCs (brief inversion + oracle) as an annotation box
    ann = []
    if "mechanism" in df.columns:
        ann.append(f"brief inversion predictor: AUROC {_auroc(lab, df['mechanism'].to_numpy()):.2f}")
    if "true_col" in df.columns:
        ann.append(f"oracle (true A columns): AUROC {_auroc(lab, df['true_col'].to_numpy()):.2f}")
    if ann:
        ax.text(0.98, -0.62, "\n".join(ann), fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#cccccc", alpha=0.9))

    ax.set_xlabel("Predicted transportability score  (cosine of predicted effects across contexts)")
    ax.set_ylabel("True continuous agreement  cos(τ_C, τ_C′)")
    ax.set_title("C4 money-shot — held-out perturbations, pooled over all four modes")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.15)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rec = os.path.join(here, "results", "c4_records.csv")
    df = pd.read_csv(rec)
    method_cols = [c for c in df.columns if c not in META]
    print(f"Loaded {len(df)} records; methods: {method_cols}")

    # Pooled AUROC + bootstrap CIs
    au_summ, gap_summ = cluster_bootstrap(df, method_cols)
    print("\n== Pooled AUROC (median [95% CI]) ==")
    for m in method_cols:
        s = au_summ[m]
        print(f"  {PRETTY.get(m, m):40s} {s['median']:.3f}  [{s['lo']:.3f}, {s['hi']:.3f}]")
    print("\n== Mechanism - null AUROC gaps (median [95% CI]) ==")
    for g, s in gap_summ.items():
        star = "  <-- CI excludes 0" if (s["lo"] > 0 or s["hi"] < 0) else ""
        print(f"  {g:28s} {s['median']:+.3f}  [{s['lo']:+.3f}, {s['hi']:+.3f}]{star}")

    # Per-mode Spearman
    sp = per_mode_spearman(df, method_cols)
    print("\n== Per-mode Spearman(score, continuous agreement) ==")
    for mode in ["none", "a", "b", "both"]:
        sub = sp[sp["mode"] == mode]
        vals = ", ".join(f"{r.method}={r.spearman:.2f}" for r in sub.itertuples() if not np.isnan(r.spearman))
        print(f"  {mode:5s}: {vals if vals else 'agreement constant (NaN)'}")

    # Assemble c4_auroc.csv
    out_rows = []
    for m in method_cols:
        s = au_summ[m]
        row = dict(method=m, scope="pooled", metric="AUROC",
                   value=round(s["median"], 4), ci_lo=round(s["lo"], 4), ci_hi=round(s["hi"], 4))
        out_rows.append(row)
    for r in sp.itertuples():
        out_rows.append(dict(method=r.method, scope=f"mode:{r.mode}", metric="Spearman(agree)",
                             value=None if np.isnan(r.spearman) else round(r.spearman, 4),
                             ci_lo=None if np.isnan(r.lo) else round(r.lo, 4),
                             ci_hi=None if np.isnan(r.hi) else round(r.hi, 4)))
    for g, s in gap_summ.items():
        out_rows.append(dict(method=g, scope="pooled", metric="AUROC_gap",
                             value=round(s["median"], 4), ci_lo=round(s["lo"], 4), ci_hi=round(s["hi"], 4)))
    out = pd.DataFrame(out_rows)
    out_path = os.path.join(here, "results", "c4_auroc.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    moneyshot(df, os.path.join(here, "results", "moneyshot.png"))
    print(f"Wrote {os.path.join(here, 'results', 'moneyshot.png')}")


if __name__ == "__main__":
    main()
