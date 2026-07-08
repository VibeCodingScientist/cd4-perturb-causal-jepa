"""
End-to-end smoke test for CausalCisTransFormer + its non-causal twin on the synthetic core.

Validates the full training/prediction path (encoder -> do-masked blocks -> delta/DE heads
-> runs/*.parquet -> core.eval), that the do-mask actually changes the model's output
(causal != non-causal), and that the model learns non-trivial, non-collapsed structure.
NOT a scientific claim about causal>non-causal — that needs the real data + full training.

Run: CD4_DATA_ROOT=$(mktemp -d) .venv/bin/python tests/test_causal_synthetic.py
"""
import os
import sys
import tempfile
import pathlib

os.environ.setdefault("CD4_DATA_ROOT", tempfile.mkdtemp(prefix="cd4-causal-"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core import contract as C

_SB = pathlib.Path(os.environ["CD4_DATA_ROOT"])
C.SPLIT_MANIFEST = _SB / "split_manifest.json"
C.HVG_LIST_PATH = _SB / "split" / "hvg_3000.txt"
C.RESULTS_DIR = _SB / "results"
C.BENCHMARK_TABLE = C.RESULTS_DIR / "benchmark_table.csv"
C.BENCHMARK_TABLE_FULL = C.RESULTS_DIR / "benchmark_table_full.csv"
C.FIGURES_DIR = _SB / "figures"

from core import eval as ev
from core import synthetic
from core.models import causal_cistransformer as cc


def test_causal_and_twin_run():
    synthetic.write_synthetic(n_genes=200, n_perts=120, seed=0)

    cfg = cc.CausalConfig(
        d_model=64, n_layers=2, n_heads=4, epochs=12, batch_size=32,
        gene_window=1000, bf16=False, grad_checkpoint=False,
    )
    cc.run_causal(splits=(C.SPLIT_CONDITION, C.SPLIT_GENE), cfg=cfg, record=True)
    cc.run_noncausal(splits=(C.SPLIT_CONDITION, C.SPLIT_GENE), cfg=cfg, record=True)

    # runs written
    for m in (C.MODEL_CAUSAL, C.MODEL_NONCAUSAL):
        for s in (C.SPLIT_CONDITION, C.SPLIT_GENE):
            assert C.run_path(m, s).exists(), (m, s)

    causal_pred = pd.read_parquet(C.run_path(C.MODEL_CAUSAL, C.SPLIT_CONDITION))
    noncausal_pred = pd.read_parquet(C.run_path(C.MODEL_NONCAUSAL, C.SPLIT_CONDITION))

    # the do-mask must change the model's behavior
    common = causal_pred.index.intersection(noncausal_pred.index)
    diff = (causal_pred.loc[common] - noncausal_pred.loc[common]).abs().to_numpy().max()
    assert diff > 1e-4, "causal and non-causal predictions are identical (mask had no effect)"

    # both produce finite, non-collapsed headline metrics on the condition hold-out
    mc = ev.evaluate(causal_pred, C.SPLIT_CONDITION)
    mn = ev.evaluate(noncausal_pred, C.SPLIT_CONDITION)
    for m, name in ((mc, "causal"), (mn, "noncausal")):
        assert np.isfinite(m[C.METRIC_PEARSON_DELTA]), name
        assert np.isfinite(m[C.METRIC_PERTURBENCH_RANK]), name
    # causal learns real per-perturbation structure (not a scientific claim vs non-causal —
    # that needs the real data + full training; here we only require genuine signal and that
    # it discriminates better than a pure mean-collapse).
    assert mc[C.METRIC_PEARSON_DELTA] > 0.1, mc[C.METRIC_PEARSON_DELTA]
    assert mc[C.METRIC_PERTURBENCH_RANK] < 0.5, mc[C.METRIC_PERTURBENCH_RANK]

    # benchmark table has all four rows recorded
    bt = pd.read_csv(C.BENCHMARK_TABLE)
    got = set(zip(bt["model"], bt["split"]))
    assert {("causal", "condition"), ("noncausal", "condition"),
            ("causal", "gene"), ("noncausal", "gene")} <= got

    print("PASS  causal cond pearson=%.3f rank=%.3f | noncausal cond pearson=%.3f rank=%.3f | "
          "|Δpred|max=%.3f" % (
              mc[C.METRIC_PEARSON_DELTA], mc[C.METRIC_PERTURBENCH_RANK],
              mn[C.METRIC_PEARSON_DELTA], mn[C.METRIC_PERTURBENCH_RANK], diff))


if __name__ == "__main__":
    test_causal_and_twin_run()
    print("\nCAUSAL MODEL SMOKE TEST PASSED")
