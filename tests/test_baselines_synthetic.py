"""
Ridge baseline on synthetic data: exercises the full baseline path (feature assembly ->
fit -> predict per split -> write runs/*.parquet -> score via core.eval) and checks that,
when gene embeddings carry real signal, Ridge learns a non-trivial, non-collapsed
predictor and beats a mean-delta predictor.

Run: CD4_DATA_ROOT=$(mktemp -d) python3.12 tests/test_baselines_synthetic.py
"""
import os
import sys
import tempfile
import pathlib

os.environ.setdefault("CD4_DATA_ROOT", tempfile.mkdtemp(prefix="cd4-base-"))
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
from core.models import baselines


def test_ridge_learns_and_records():
    synthetic.write_synthetic(n_genes=200, n_perts=120, seed=0)

    baselines.run_ridge(splits=(C.SPLIT_CONDITION, C.SPLIT_GENE), record=True)

    # runs written for both splits
    assert C.run_path(C.MODEL_RIDGE, C.SPLIT_CONDITION).exists()
    assert C.run_path(C.MODEL_RIDGE, C.SPLIT_GENE).exists()

    # score condition hold-out directly and compare to a mean predictor
    gt = ev.ground_truth(C.SPLIT_CONDITION)
    ridge_pred = pd.read_parquet(C.run_path(C.MODEL_RIDGE, C.SPLIT_CONDITION))
    ridge = ev.evaluate(ridge_pred, C.SPLIT_CONDITION, truth=gt)
    mean_pred = pd.DataFrame(np.tile(ev.ground_truth(C.SPLIT_CONDITION).mean(axis=0).to_numpy(),
                                     (len(gt), 1)), index=gt.index, columns=gt.columns)
    mean = ev.evaluate(mean_pred, C.SPLIT_CONDITION, truth=gt)

    assert ridge[C.METRIC_PEARSON_DELTA] > 0.4, ridge[C.METRIC_PEARSON_DELTA]
    assert ridge[C.METRIC_PEARSON_DELTA] > mean[C.METRIC_PEARSON_DELTA]
    assert ridge[C.MODE_COLLAPSE_FLAG] is False  # Ridge discriminates perturbations

    # gene hold-out (interpolation over unseen genes) should also carry signal
    ridge_gene = ev.evaluate(
        pd.read_parquet(C.run_path(C.MODEL_RIDGE, C.SPLIT_GENE)), C.SPLIT_GENE)
    assert ridge_gene[C.METRIC_PEARSON_DELTA] > 0.2, ridge_gene[C.METRIC_PEARSON_DELTA]

    # benchmark table well-formed with the ridge rows recorded
    bt = pd.read_csv(C.BENCHMARK_TABLE)
    assert set(bt["split"]) >= {"condition", "gene"}
    assert (bt["model"] == "ridge").all()

    print("PASS  ridge cond pearson=%.3f (mean=%.3f, collapsed=%s)  gene pearson=%.3f" % (
        ridge[C.METRIC_PEARSON_DELTA], mean[C.METRIC_PEARSON_DELTA],
        mean[C.MODE_COLLAPSE_FLAG], ridge_gene[C.METRIC_PEARSON_DELTA]))


if __name__ == "__main__":
    test_ridge_learns_and_records()
    print("\nRIDGE BASELINE TEST PASSED")
