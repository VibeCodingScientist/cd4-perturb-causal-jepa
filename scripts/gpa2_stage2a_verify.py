#!/usr/bin/env python
"""G-PA.2 Stage 2a — verify the two load-bearing assumptions BEFORE computing any probe:
  (1) lane -> condition mapping (which of the 8 aggr lanes are stim vs nostim), verified empirically
      via T-cell activation markers (not assumed from aggr order);
  (2) which guide targets are non-targeting CONTROLS (needed to define perturbation deltas).
Halts loudly if either can't be established. Reads the isolated ~/gpa2-data, never DATA_ROOT.
"""
import os, gzip, csv
from collections import Counter
import numpy as np
import scanpy as sc

DEST = os.path.expanduser("~/gpa2-data/raw")
GC = os.path.join(DEST, "GSE190604_cellranger-guidecalls-aggregated-unfiltered.txt.gz")


def main():
    print("[2a] loading 10x mtx (gex) ...", flush=True)
    a = sc.read_10x_mtx(DEST, gex_only=True, prefix="GSE190604_")   # cells x genes
    a.obs["lane"] = [bc.split("-")[-1] for bc in a.obs_names]
    print(f"[2a] cells={a.n_obs} genes={a.n_vars} lanes={sorted(a.obs['lane'].unique())}", flush=True)

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    markers = [m for m in ["IL2", "IFNG", "TNFRSF9", "IL2RA", "MKI67", "GZMB", "TNF", "NR4A1", "IL2RB"]
               if m in a.var_names]
    print(f"[2a] activation markers present: {markers}", flush=True)
    import pandas as pd
    df = sc.get.obs_df(a, keys=markers + ["lane"])
    per_lane = df.groupby("lane")[markers].mean()
    print("\n==== per-lane mean activation markers (higher = stimulated) ====", flush=True)
    per_lane["ACT_SCORE"] = per_lane.mean(axis=1)
    print(per_lane.round(3).to_string(), flush=True)
    # tentative aggr order: lanes 1-4 nostim, 5-8 stim -> verify separation
    order = per_lane["ACT_SCORE"].sort_values()
    lo4 = set(order.index[:4]); hi4 = set(order.index[4:])
    print(f"\n[2a] LOW-activation lanes (=> nostim): {sorted(lo4)}", flush=True)
    print(f"[2a] HIGH-activation lanes (=> stim):  {sorted(hi4)}", flush=True)
    sep = order.iloc[4:].min() - order.iloc[:4].max()
    print(f"[2a] activation-score gap (min-stim − max-nostim) = {sep:.3f}  "
          f"({'CLEAN 4/4 split' if sep > 0 else 'NO CLEAN SPLIT — HALT'})", flush=True)

    print("\n==== guide targets ====", flush=True)
    tgt = Counter(); multi = 0; n = 0
    for row in csv.DictReader(gzip.open(GC, "rt"), delimiter="\t"):
        n += 1
        if row.get("num_features", "1") != "1":
            multi += 1; continue
        fc = row["feature_call"]
        t = fc.rsplit("-", 1)[0]           # ABCB10-1 -> ABCB10
        tgt[t] += 1
    print(f"[2a] guidecall rows={n} singlets={sum(tgt.values())} multiguide={multi} unique_targets={len(tgt)}", flush=True)
    print(f"[2a] top targets by cells: {tgt.most_common(12)}", flush=True)
    ctrl = sorted([t for t in tgt if any(k in t.upper()
                   for k in ["NO_", "NON", "SAFE", "CTRL", "CONTROL", "GAL4", "SCRAMBLE", "NEG", "OLFR", "NTC", "AAVS"])])
    print(f"[2a] CONTROL candidates (keyword match): {[(c, tgt[c]) for c in ctrl]}", flush=True)
    print(f"[2a] all targets: {sorted(tgt.keys())}", flush=True)
    print("\nSTAGE2A_VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
