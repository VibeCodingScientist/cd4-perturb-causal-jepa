#!/usr/bin/env python
"""C2 positive control on the RESTORED DATA_ROOT — prove the restored data is real, not a
schema-matched fixture, by reproducing a KNOWN ANSWER (the committed within-distribution C2).

After the synthetic-contamination event, an "aligned to the HVG panel" check is only a shape check.
This is the reality check: re-train causal + non-causal on the restored data and reproduce
`results/benchmark_table.csv`'s within-distribution C2 = causal − non-causal Pearson-δ(top-50 DEG):
  committed: condition 0.3436 − 0.2255 = +0.118 ; gene 0.3675 − 0.2056 = +0.162.

Full-train reproduction (no external-reg holdout): this is the EXACT known answer, and data-reality is a
property of the data, not of which 10 perts G-F.2 held out — the G-F.2 run used this same restored
pseudobulk (minus 10 negligible perts) + this same ESM2. If C2 reproduces here, the G-F.2 data is real.

ISOLATED: reuses cc internals (_build_samples/_train/_predict_split) but NEVER calls _run — writes only
results/c2_control.csv + results/C2_STATUS.txt. Does NOT touch runs/ or benchmark_table.csv.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from core import contract as C
from core import eval as ev
from core import split as split_mod
from core.models import causal_cistransformer as cc

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
STATUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "C2_STATUS.txt")

# committed truth (results/benchmark_table.csv), verified this run
COMMITTED = {
    "condition": {"causal": 0.3436, "noncausal": 0.2255, "c2": 0.1180},
    "gene":      {"causal": 0.3675, "noncausal": 0.2056, "c2": 0.1619},
}
TOL = 0.06  # generous band for retraining/GPU-nondeterminism stochasticity


def train_and_score(use_causal, cfg, device):
    import torch
    train_pb = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    hvg = [g for g in split_mod.load_hvg() if g in C.pseudobulk_expr(train_pb).columns]
    samples = cc._build_samples(train_pb, hvg, cfg)
    esm2_t, ctx_t = cc._feature_tensors(hvg, device)
    torch.manual_seed(cfg.seed)
    model = cc._build_model(cfg, use_causal).to(device)
    t0 = time.time()
    cc._train(model, samples, esm2_t, ctx_t, cfg, device)
    scores = {}
    for split in (C.SPLIT_CONDITION, C.SPLIT_GENE):
        pred = cc._predict_split(model, split, hvg, esm2_t, ctx_t, cfg, device)
        scores[split] = float(ev.evaluate(pred, split, full=False)[C.METRIC_PEARSON_DELTA])
    print(f"[c2] {'causal' if use_causal else 'noncausal'} trained {(time.time()-t0)/60:.1f} min  scores={scores}", flush=True)
    return scores


def main():
    import torch
    open(STATUS, "w").write("RUNNING\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cc.CausalConfig(epochs=40)   # committed CP2 config
    print(f"[c2] device={device} epochs={cfg.epochs}  full-train reproduction on restored DATA_ROOT", flush=True)

    causal = train_and_score(True, cfg, device)
    noncausal = train_and_score(False, cfg, device)

    rows, ok = [], True
    for name, s in (("condition", C.SPLIT_CONDITION), ("gene", C.SPLIT_GENE)):
        cval, nval = causal[s], noncausal[s]
        c2 = cval - nval
        com = COMMITTED[name]
        pass_split = (abs(cval - com["causal"]) < TOL and abs(nval - com["noncausal"]) < TOL
                      and abs(c2 - com["c2"]) < TOL and cval > nval)
        ok = ok and pass_split
        rows.append(dict(split=name, causal=round(cval, 4), noncausal=round(nval, 4), c2=round(c2, 4),
                         committed_causal=com["causal"], committed_noncausal=com["noncausal"],
                         committed_c2=com["c2"], within_tol=pass_split))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "c2_control.csv"), index=False)
    print("\n########## C2 POSITIVE CONTROL ##########", flush=True)
    print(df.to_string(index=False), flush=True)
    verdict = "PASS" if ok else "FAIL"
    print(f"\nC2 CONTROL VERDICT: {verdict}  (tol=±{TOL}; committed condition +0.118 / gene +0.162)", flush=True)
    print("Interpretation: PASS => restored data is REAL => G-F.2 numbers trustworthy; "
          "FAIL => restore suspect => G-F.2 not interpretable, suspend the 7th negative.", flush=True)
    open(STATUS, "w").write(f"{verdict}\n")
    print("C2_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
