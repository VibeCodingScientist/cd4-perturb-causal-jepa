#!/usr/bin/env python
"""C-BUDGET Stage 3 — per-pert decomposition, do-operator localization, ceiling figure.

CPU-only, seconds. Consumes results/budget_reliability.csv, fraction_of_ceiling.csv,
budget_cross_donor.csv (Stages 1-2) + the committed runs/. Emits:
  results/budget_decomposition.csv   per-pert A/B/C + model fraction-of-ceiling
  results/do_operator_localization.csv  is the do-operator's edge (C2) on reliable perts?
  figures/budget_ceiling.png         the stacked A|C|B budget per context, models located

Bucket definitions (fraction of TOTAL top-50 response variance):
  B  = irreducible noise floor        = 1 - r_ceiling                          [IN-PROJECT, Stage 1]
  A  = linear-explainable             = frac_of_ceiling(ridge, fitted-linear) * r_ceiling
                                        (predictive, held-out; upper bound on linear)  [IN-PROJECT]
  C  = structured residual            = r_ceiling - A   (what NO linear model captures) [INFERENCE]
CIPHER Sigma-u (mechanistic-linear, Dev 4 committed R2~0.30) is shown as a reference tick,
NOT used to define A (avoids naive 1-R2). Never touches CP2 or Dev 4's artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import contract as C
from core import eval as ev

# [IN-PROJECT] Dev 4's committed real-data CIPHER Sigma-u fraction of TOTAL variance (median
# R2 ~0.30). Aggregate mechanistic-linear reference only; per-split refinement pending Dev 4.
CIPHER_A_REF = 0.30
TOPK = C.TOP_DEG_N


def per_pert_pearson(pred, truth):
    genes = [g for g in truth.columns if g in pred.columns]
    perts = [p for p in truth.index if p in pred.index]
    P = pred.loc[perts, genes].to_numpy(float)
    T = truth.loc[perts, genes].to_numpy(float)
    out = {}
    for i, p in enumerate(perts):
        tk = np.argsort(-np.abs(T[i]))[:TOPK]
        x, y = P[i, tk], T[i, tk]
        out[p] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else np.nan
    return out


def _pp(split):
    truth = ev.ground_truth(split)
    return {m: per_pert_pearson(pd.read_parquet(C.run_path(m, split)), truth)
            for m in (C.MODEL_CAUSAL, C.MODEL_NONCAUSAL, C.MODEL_RIDGE)}


def decomposition(rel):
    """Per-pert A/B/C + model fraction-of-ceiling; A from per-pert ridge frac (linear, held-out)."""
    rows = []
    for split in [C.SPLIT_CONDITION, C.SPLIT_GENE]:
        pe = _pp(split)
        rsub = rel[rel.split == split].set_index("pert_id")
        for p, r in rsub.iterrows():
            ceil = r["r_ceiling"]
            if not np.isfinite(ceil) or ceil <= 0.05:
                continue
            reliable = ceil
            ridge_frac = pe[C.MODEL_RIDGE].get(p, np.nan) / ceil
            causal_frac = pe[C.MODEL_CAUSAL].get(p, np.nan) / ceil
            A = max(0.0, min(reliable, (pe[C.MODEL_RIDGE].get(p, np.nan)))) if np.isfinite(pe[C.MODEL_RIDGE].get(p, np.nan)) else np.nan
            rows.append({
                "pert_id": p, "split": split, "signal_l2": r["signal_l2"], "n_units": r["n_units"],
                "reliable_r_ceiling": reliable, "noise_floor_B": 1 - reliable,
                "linear_A_ridge": A, "structured_C": (reliable - A) if np.isfinite(A) else np.nan,
                "r_ceiling_specific": r["r_ceiling_specific"], "cross_donor_r": r["cross_donor_r"],
                "ridge_frac_of_ceiling": ridge_frac, "causal_frac_of_ceiling": causal_frac,
            })
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "budget_decomposition.csv", index=False)
    print(f"[decomp] wrote budget_decomposition.csv ({len(df)} perts)")
    return df


def localize_do_operator(rel):
    rows = []
    for split in [C.SPLIT_CONDITION, C.SPLIT_GENE]:
        pe = _pp(split)
        rc = rel[rel.split == split].set_index("pert_id")["r_ceiling"].to_dict()
        perts = [p for p in pe[C.MODEL_CAUSAL] if p in rc and np.isfinite(rc[p])]
        gap = np.array([pe[C.MODEL_CAUSAL][p] - pe[C.MODEL_NONCAUSAL][p] for p in perts])   # C2 per pert
        gap_vs_ridge = np.array([pe[C.MODEL_CAUSAL][p] - pe[C.MODEL_RIDGE][p] for p in perts])
        rcv = np.array([rc[p] for p in perts])
        finite = np.isfinite(gap) & np.isfinite(rcv)
        gap, gap_vs_ridge, rcv = gap[finite], gap_vs_ridge[finite], rcv[finite]
        med = np.median(rcv)
        hi, lo = rcv >= med, rcv < med
        corr = float(np.corrcoef(gap, rcv)[0, 1]) if gap.std() > 1e-9 else np.nan
        rows.append({
            "split": split, "n": int(finite.sum()),
            "c2_gap_median": float(np.median(gap)),
            "c2_gap_reliable_perts": float(np.median(gap[hi])),
            "c2_gap_unreliable_perts": float(np.median(gap[lo])),
            "corr_gap_vs_reliability": corr,
            "causal_minus_ridge_median": float(np.median(gap_vs_ridge)),
        })
        print(f"[localize] {split}: C2 median={np.median(gap):+.4f}  reliable-perts={np.median(gap[hi]):+.4f}  "
              f"unreliable={np.median(gap[lo]):+.4f}  corr(C2,r_ceiling)={corr:+.3f}  "
              f"causal-ridge={np.median(gap_vs_ridge):+.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "do_operator_localization.csv", index=False)
    return df


def figure(rel, frac):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contexts = [(C.SPLIT_CONDITION, "Condition hold-out\n(zero-shot Stim48hr)"),
                (C.SPLIT_GENE, "Gene hold-out\n(unseen targets)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    C_A, C_C, C_NOISE = "#0d8b96", "#e2a13b", "#c9ccd1"
    for ax, (split, title) in zip(axes, contexts):
        s = rel[rel.split == split]
        B = float(s.noise_floor.median())
        reliable = 1 - B
        fsub = frac[frac.split == split].set_index("model")["frac_of_ceiling_median"]
        ridge_frac = float(fsub.get("ridge", 0.0))
        A = max(0.0, ridge_frac) * reliable                 # fitted-linear-explainable (of total)
        Cbar = max(0.0, reliable - A)
        ax.bar(0, A, 0.5, color=C_A, label="A · linear-explainable (Ridge, held-out)")
        ax.bar(0, Cbar, 0.5, bottom=A, color=C_C, label="C · structured residual (no linear model)")
        ax.bar(0, B, 0.5, bottom=A + Cbar, color=C_NOISE, label="B · irreducible noise floor")
        for y0, h, lab, col in [(0, A, f"A≈{A:.2f}", "white"),
                                (A, Cbar, f"C≈{Cbar:.2f}", "white"),
                                (A + Cbar, B, f"B≈{B:.2f}", "#444")]:
            if h > 0.04:
                ax.text(0, y0 + h / 2, lab, ha="center", va="center", fontsize=10,
                        color=col, fontweight="bold" if col == "white" else "normal")
        # do-operator fraction-of-ceiling -> fraction of TOTAL variance (frac x reliable).
        # causal and jepa_causal coincide (JEPA-init neutral on this axis) -> one line.
        if "causal" in fsub.index:
            y = fsub["causal"] * reliable
            ax.plot([-0.33, 0.33], [y, y], color="#b5179e", lw=2.2)
            ax.text(0.37, y, f"do-operator (causal): {fsub['causal']:.2f}×ceil", va="center",
                    fontsize=8.5, color="#b5179e")
        ax.plot([-0.33, 0.33], [reliable, reliable], color="#111", lw=1.4, ls="--")
        ax.text(0.37, reliable, "ceiling (reliable)", va="center", fontsize=8.5, color="#111")
        ax.plot([-0.25, 0.25], [CIPHER_A_REF, CIPHER_A_REF], color="#0d8b96", lw=1.1, ls=":")
        ax.text(0.37, CIPHER_A_REF, f"CIPHER Σu ref: {CIPHER_A_REF:.2f}", va="center", fontsize=7.5, color="#0d8b96")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.6, 1.7); ax.set_ylim(0, 1.02); ax.set_xticks([])
    axes[0].set_ylabel("Response predictability — Pearson-δ scale (top-50 DEGs)\n1.0 = perfect prediction on noise-free truth;  A+C+B = 1")
    axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    fig.suptitle("Predictability budget — real activated CD4⁺ CRISPRi response", fontsize=13)
    fig.tight_layout()
    out = C.FIGURES_DIR / "budget_ceiling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    C.ensure_dirs()
    rel = pd.read_csv(C.RESULTS_DIR / "budget_reliability.csv")
    frac = pd.read_csv(C.RESULTS_DIR / "fraction_of_ceiling.csv")
    decomposition(rel)
    localize_do_operator(rel)
    figure(rel, frac)
    print("\n=== Stage 3 wrote budget_decomposition.csv + do_operator_localization.csv + budget_ceiling.png ===")


if __name__ == "__main__":
    main()
