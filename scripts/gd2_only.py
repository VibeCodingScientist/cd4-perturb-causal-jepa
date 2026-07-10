#!/usr/bin/env python
"""Isolated G-D.2 (memory-light): frees delta/s_perp, casts s to float32, then runs gd2_recovery."""
import sys, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core import split as split_mod
from scripts.donor_gate import load_guide_effects, gd2_recovery, RES

hvg = list(split_mod.load_hvg())
meta, s, s_perp, delta, donors = load_guide_effects(hvg)
del s_perp, delta; gc.collect()
s = np.ascontiguousarray(s, dtype=np.float32)
print(f"[gd2-only] freed; s={s.shape} {s.dtype}", flush=True)
d2 = gd2_recovery(meta, s, donors, np.random.default_rng(42))
med = d2[["cond_pred", "avg_pred", "mean_pred", "perm_pred"]].median()
print(f"  n pairs={len(d2)}")
print(f"  donor-CONDITIONED (same-donor g1->g2)   = {med.cond_pred:.4f}")
print(f"  donor-AVERAGED (cross-donor consensus)  = {med.avg_pred:.4f}")
print(f"  predict-the-donor-MEAN                  = {med.mean_pred:.4f}")
print(f"  donor-PERMUTED (wrong-donor g1)         = {med.perm_pred:.4f}")
gd2_go = (med.cond_pred >= 0.20 and med.cond_pred > med.avg_pred + 0.05
          and med.cond_pred > med.mean_pred + 0.05 and med.cond_pred > med.perm_pred + 0.05)
print(f"  G-D.2 VERDICT: {'GO' if gd2_go else 'NO-GO'}")
d2.to_csv(RES / "donor_structure_gd2_pairs.csv", index=False)
med.to_frame("median").to_csv(RES / "donor_structure_gd2_medians.csv")
print("GD2_DONE")
