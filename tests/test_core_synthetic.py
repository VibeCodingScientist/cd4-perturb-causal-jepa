"""
End-to-end smoke test of the Lane-C core on synthetic data (no torch / anndata).

Exercises the REAL code paths: pseudobulk deltas + train/test routing, DEG-frequency,
split freeze/verify, and the frozen eval harness including the mode-collapse detector.

Run standalone:  CD4_DATA_ROOT=$(mktemp -d) python3.12 tests/test_core_synthetic.py
Or with pytest:  CD4_DATA_ROOT=$(mktemp -d) pytest tests/test_core_synthetic.py -q
"""
import os
import sys
import tempfile

# Sandbox DATA_ROOT to a temp dir BEFORE importing core (contract reads the env at import).
os.environ.setdefault("CD4_DATA_ROOT", tempfile.mkdtemp(prefix="cd4-synth-"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pathlib

import numpy as np
import pandas as pd

from core import contract as C

# Redirect the COMMITTED-artifact paths (split manifest, HVG list, benchmark tables,
# figures) into the sandbox so the test never mutates the real repo files. Other core
# modules read these as C.<attr> at call time, so patching the module attributes here
# is sufficient. (DATA_ROOT-based paths are already sandboxed via CD4_DATA_ROOT.)
_SB = pathlib.Path(os.environ["CD4_DATA_ROOT"])
C.SPLIT_MANIFEST = _SB / "split_manifest.json"
C.HVG_LIST_PATH = _SB / "split" / "hvg_3000.txt"
C.RESULTS_DIR = _SB / "results"
C.BENCHMARK_TABLE = C.RESULTS_DIR / "benchmark_table.csv"
C.BENCHMARK_TABLE_FULL = C.RESULTS_DIR / "benchmark_table_full.csv"
C.FIGURES_DIR = _SB / "figures"

from core import eval as ev
from core import split as split_mod
from core import synthetic


def _build():
    return synthetic.write_synthetic(n_genes=200, n_perts=120, seed=0)


def _mean_predictor(truth: pd.DataFrame) -> pd.DataFrame:
    """A degenerate model that predicts the SAME (mean) delta for every perturbation."""
    mean_row = truth.mean(axis=0)
    return pd.DataFrame(
        np.tile(mean_row.to_numpy(), (len(truth), 1)),
        index=truth.index, columns=truth.columns,
    )


def _oracle_predictor(truth: pd.DataFrame, noise: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return truth + rng.normal(0, noise, truth.shape)


def test_core_end_to_end():
    art = _build()
    man = art["manifest"]

    # --- split froze correctly ---
    assert man.data_frozen
    split_mod.verify()  # data_frozen check passes
    n_hold = len(man.gene_holdout)
    assert n_hold == round(120 * C.GENE_HOLDOUT_FRACTION), n_hold  # 18

    # --- pseudobulk schema + routing ---
    train, test = art["train"], art["test"]
    assert isinstance(train.columns, pd.MultiIndex)
    assert set(train.columns.get_level_values(0).unique()) == {"expr", "delta"}
    # no Stim48hr, no held-out gene, no donor_4 leaked into train
    tr_pert = train.index.get_level_values("pert_id")
    tr_cond = train.index.get_level_values("condition")
    tr_donor = train.index.get_level_values("donor")
    assert (tr_cond != "Stim48hr").all()
    assert not set(tr_pert).intersection(man.gene_holdout)
    assert (tr_donor != C.DONOR_PROBE).all()

    # --- ground truth for both primary splits ---
    gt_cond = ev.ground_truth(C.SPLIT_CONDITION, pseudobulk_test=test)
    gt_gene = ev.ground_truth(C.SPLIT_GENE, pseudobulk_test=test)
    assert gt_cond.shape[0] > 0 and gt_gene.shape[0] == n_hold
    assert C.CONTROL_PERT_ID not in gt_cond.index

    # --- eval: an oracle-ish model scores well and is NOT flagged collapsed ---
    good = ev.evaluate(_oracle_predictor(gt_cond, noise=0.15), C.SPLIT_CONDITION, truth=gt_cond)
    assert good[C.METRIC_PEARSON_DELTA] > 0.8, good[C.METRIC_PEARSON_DELTA]
    assert good[C.METRIC_DES] > 0.8, good[C.METRIC_DES]
    assert good[C.METRIC_PERTURBENCH_RANK] < 0.2, good[C.METRIC_PERTURBENCH_RANK]
    assert good[C.MODE_COLLAPSE_FLAG] is False

    # --- eval: the mean predictor is caught by the mode-collapse detector ---
    collapsed = ev.evaluate(_mean_predictor(gt_cond), C.SPLIT_CONDITION, truth=gt_cond)
    assert 0.4 <= collapsed[C.METRIC_PERTURBENCH_RANK] <= 0.6, collapsed[C.METRIC_PERTURBENCH_RANK]
    assert collapsed[C.MODE_COLLAPSE_FLAG] is True
    # ... and it should score far worse on delta accuracy than the oracle
    assert collapsed[C.METRIC_PEARSON_DELTA] < good[C.METRIC_PEARSON_DELTA]

    # --- recording writes a well-formed benchmark row ---
    ev.evaluate_and_record(_oracle_predictor(gt_cond, 0.15), C.SPLIT_CONDITION, C.MODEL_RIDGE, truth=gt_cond)
    bt = pd.read_csv(C.BENCHMARK_TABLE)
    assert list(bt.columns) == list(C.BENCHMARK_COLUMNS)
    assert ((bt["model"] == "ridge") & (bt["split"] == "condition")).any()

    print("PASS  pearson_delta(good)=%.3f  rank(good)=%.3f  rank(collapsed)=%.3f" % (
        good[C.METRIC_PEARSON_DELTA], good[C.METRIC_PERTURBENCH_RANK],
        collapsed[C.METRIC_PERTURBENCH_RANK]))


if __name__ == "__main__":
    test_core_end_to_end()
    print("\nALL CORE SMOKE TESTS PASSED")
