"""Supplementary: how does the mechanism-vs-null gap move with sequencing depth (n_cells)?

A central caveat of the C4 result is that in this linear-Gaussian simulator the control-cell
covariance is a near-sufficient statistic for the causal matrix A, so the correlation nulls
sharpen FASTER than the sparse interventional A-estimate as cells increase. This script makes
that explicit on the reported seeds.

Outputs: results/sensitivity_ncells.csv  and  results/sensitivity_ncells.png
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_c4 import CONFIG, generate

N_CELLS = [500, 1000, 2000, 4000]
METHODS = ["mechanism", "mechanism_col", "corr_null", "gears_null", "true_col"]
LABELS = {"mechanism": "Mechanism (inversion)", "mechanism_col": "Mechanism (column-k)",
          "corr_null": "Null: correlation", "gears_null": "Null: co-expression",
          "true_col": "Oracle (true A)"}
COLORS = {"mechanism": "#7fb0d3", "mechanism_col": "#1f77b4", "corr_null": "#8a9099",
          "gears_null": "#c2c7cd", "true_col": "#2ca02c"}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for nc in N_CELLS:
        cfg = dict(CONFIG); cfg["n_cells"] = nc
        df = generate(cfg, verbose=False)
        lab = df["label"].to_numpy()
        rec = {"n_cells": nc}
        for m in METHODS:
            rec[m] = round(float(roc_auc_score(lab, df[m].to_numpy())), 4)
        rows.append(rec)
        print(f"n_cells={nc:5d} | " + "  ".join(f"{m}={rec[m]:.3f}" for m in METHODS))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(here, "results", "sensitivity_ncells.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for m in METHODS:
        ax.plot(out["n_cells"], out[m], marker="o", color=COLORS[m], label=LABELS[m],
                lw=2 if m in ("mechanism_col", "corr_null") else 1.4,
                ls="-" if "null" not in m and m != "true_col" else ("--" if "null" in m else ":"))
    ax.axhline(0.5, color="black", lw=0.8, alpha=0.4)
    ax.text(N_CELLS[0], 0.51, "chance", fontsize=8, alpha=0.6)
    ax.set_xscale("log"); ax.set_xticks(N_CELLS); ax.set_xticklabels(N_CELLS)
    ax.set_xlabel("cells per (context, perturbation)")
    ax.set_ylabel("pooled AUROC (transportable vs blocked)")
    ax.set_title("C4 sensitivity — nulls sharpen faster than the mechanism with depth")
    ax.set_ylim(0.45, 1.02); ax.grid(True, alpha=0.15); ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(here, "results", "sensitivity_ncells.png"), dpi=150)
    print("Wrote results/sensitivity_ncells.{csv,png}")


if __name__ == "__main__":
    main()
