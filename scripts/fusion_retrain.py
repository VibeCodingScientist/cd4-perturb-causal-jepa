#!/usr/bin/env python
"""C-FUSE 1b — re-train the do-operator + twin with external regulators held out, then predict them.

Reuses the exact CP2 training recipe (core.models.causal_cistransformer) but removes the external
regulators (Weinstock 2024 + Freimer 2022, mapped to ENSG, that are in the HVG panel so the token model
can represent them) from the TRAIN pseudobulk. So the retrained model predicts those regulators'
effects ZERO-SHOT (never trained on them) -> a genuinely non-circular external-validation test.

Isolated: reads committed train pseudobulk + features from DATA_ROOT (read-only), writes predictions to
results/fusion_pred_{causal,noncausal}.parquet. Never touches the frozen runs/ or the frozen split.
Args: [epochs]  (default 40 = CP2 config; pass 1 for a timing probe).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from core import contract as C
from core import split as split_mod
from core.models import causal_cistransformer as cc

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def holdout_regulators(hvg):
    """External regulators (from the measurable-edge table) that are in HVG = the predictable/testable
    held-out set. Returns their ENSG set."""
    m = pd.read_csv(os.path.join(RES, "fusion_measurable_edges.csv"))
    hvgs = set(hvg)
    regs = sorted(set(r for r in m.reg_ens.unique() if r in hvgs))
    return regs


def predict_regulators(model, regs, hvg, esm2_t, ctx_t, cfg, device):
    """Predict Δ (over HVG) for a specific list of regulator perturbations (all in HVG)."""
    import torch
    gpos = {g: i for i, g in enumerate(hvg)}
    train = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    ctrl = C.pseudobulk_expr(train)[hvg].xs(C.CONTROL_PERT_ID, level="pert_id").mean(axis=0).to_numpy(float)
    chunk = [p for p in regs if p in gpos]
    vals = np.tile(ctrl, (len(chunk), 1)); pcols = []
    for j, p in enumerate(chunk):
        pi = gpos[p]; vals[j, pi] = ctrl[pi] * (1.0 - cfg.knockdown_frac); pcols.append(pi)
    model.eval()
    with torch.no_grad():
        vt = torch.tensor(vals, dtype=torch.float32, device=device)
        pidx = torch.tensor(pcols, dtype=torch.long, device=device)
        dhat, _ = model(vt, esm2_t, ctx_t, pidx)
        arr = dhat.float().cpu().numpy()
    return pd.DataFrame(arr, index=chunk, columns=hvg)


def main():
    import torch
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[retrain] device={device} epochs={epochs}", flush=True)
    cfg = cc.CausalConfig(epochs=epochs)

    hvg = split_mod.load_hvg()
    train_pb = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    hvg = [g for g in hvg if g in C.pseudobulk_expr(train_pb).columns]
    regs = holdout_regulators(hvg)
    print(f"[retrain] holding out {len(regs)} external regulators (in HVG): "
          f"{regs[:12]}{'...' if len(regs) > 12 else ''}", flush=True)

    # remove the external regulators from the TRAIN pseudobulk (any level 'pert_id' rows)
    idx = train_pb.index
    plevel = idx.get_level_values("pert_id") if isinstance(idx, pd.MultiIndex) else train_pb["pert_id"]
    keep = ~pd.Index(plevel).isin(regs)
    train_rh = train_pb[np.asarray(keep)]
    n_before = pd.Index(plevel).nunique(); n_after = pd.Index(train_rh.index.get_level_values("pert_id")).nunique()
    print(f"[retrain] train perts: {n_before} -> {n_after} (removed {n_before - n_after})", flush=True)

    esm2_t, ctx_t = cc._feature_tensors(hvg, device)
    samples = cc._build_samples(train_rh, hvg, cfg)
    print(f"[retrain] training samples: {samples.values.shape}", flush=True)

    for name, mask in [("causal", True), ("noncausal", False)]:
        torch.manual_seed(cfg.seed)
        model = cc._build_model(cfg, mask).to(device)
        import time; t0 = time.time()
        cc._train(model, samples, esm2_t, ctx_t, cfg, device)
        print(f"[retrain] {name} trained in {(time.time()-t0)/60:.1f} min", flush=True)
        pred = predict_regulators(model, regs, hvg, esm2_t, ctx_t, cfg, device)
        out = os.path.join(RES, f"fusion_pred_{name}.parquet")
        pred.to_parquet(out)
        print(f"[retrain] wrote {out}  shape={pred.shape}", flush=True)
    print("RETRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
