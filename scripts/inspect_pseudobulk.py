"""Sanity-check the built pseudobulk before training. Run: .venv/bin/python scripts/inspect_pseudobulk.py"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core import contract as C
from core import eval as ev
from core import split as sp


def summ(df, name):
    idx = df.index
    conds = sorted(set(idx.get_level_values("condition")))
    donors = sorted(set(idx.get_level_values("donor")))
    nperts = idx.get_level_values("pert_id").nunique()
    print("  %-6s rows=%d conds=%s donors=%s nperts=%d" % (name, len(df), conds, donors, nperts))


def main():
    man = sp.load()
    print("split: data_frozen=%s gene_holdout=%d hvg=%d sha=%s..." % (
        man.data_frozen, len(man.gene_holdout), len(sp.load_hvg()), man.sha256_h5ad[:12]))
    tr = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    te = pd.read_parquet(C.PSEUDOBULK_TEST)
    summ(tr, "train")
    summ(te, "test")

    d_tr = C.pseudobulk_delta(tr).to_numpy()
    print("train delta |.|: mean=%.4f max=%.3f  (control rows should be ~0)" % (
        np.abs(d_tr).mean(), np.abs(d_tr).max()))
    # control self-delta sanity
    dd = C.pseudobulk_delta(tr)
    ctrl = dd[dd.index.get_level_values("pert_id") == C.CONTROL_PERT_ID]
    if len(ctrl):
        print("  control-row |delta| mean=%.5f (expect ~0)" % float(np.abs(ctrl.to_numpy()).mean()))

    for split in (C.SPLIT_CONDITION, C.SPLIT_GENE):
        gt = ev.ground_truth(split)
        ep = ev.evaluable_perts(split)
        print("%s: GT perts=%d  evaluable(HVG-panel) perts=%d" % (split, len(gt), len(ep)))

    print("deg_freq:", C.DEG_FREQ_CACHE.exists(),
          pd.read_parquet(C.DEG_FREQ_CACHE).shape if C.DEG_FREQ_CACHE.exists() else None)


if __name__ == "__main__":
    main()
