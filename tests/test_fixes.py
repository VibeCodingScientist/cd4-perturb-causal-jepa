"""
Regression tests for the adversarial-review findings (so they can't silently return).

Run: CD4_DATA_ROOT=$(mktemp -d) python3.12 tests/test_fixes.py
"""
import os
import sys
import tempfile
import pathlib

os.environ.setdefault("CD4_DATA_ROOT", tempfile.mkdtemp(prefix="cd4-fixes-"))
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

from core import data, eval as ev, features as feat, synthetic
from core.pseudobulk import PseudobulkAccumulator


def test_accumulator_groups_correctly():  # finding #1 (critical)
    obs = pd.DataFrame({
        "pert_id": ["p1", "p1", C.CONTROL_PERT_ID, "p2"],
        "condition": ["Rest"] * 4,
        "donor": ["donor_1"] * 4,
    })
    X = np.arange(16, dtype=float).reshape(4, 4)
    acc = PseudobulkAccumulator(["g0", "g1", "g2", "g3"])
    acc.add(obs, X)            # must NOT raise (was KeyError on list-of-tuples groupby)
    res = acc.result()
    assert res.loc[("p1", "Rest", "donor_1")].tolist() == [2, 3, 4, 5]      # mean of rows 0,1
    assert res.loc[(C.CONTROL_PERT_ID, "Rest", "donor_1")].tolist() == [8, 9, 10, 11]
    # two chunks accumulate (streaming) correctly
    acc2 = PseudobulkAccumulator(["g0", "g1"])
    acc2.add(pd.DataFrame({"pert_id": ["p1"], "condition": ["Rest"], "donor": ["d1"]}), np.array([[2.0, 4.0]]))
    acc2.add(pd.DataFrame({"pert_id": ["p1"], "condition": ["Rest"], "donor": ["d1"]}), np.array([[4.0, 8.0]]))
    assert acc2.result().loc[("p1", "Rest", "d1")].tolist() == [3.0, 6.0]
    print("PASS  #1 PseudobulkAccumulator.add groups + streams correctly")


def test_czi_obs_mapping():  # finding-adjacent: CZI adapter correctness
    obs = pd.DataFrame({
        "perturbed_gene_id": ["ENSG1", "ENSG2", "ENSG9"],
        "culture_condition": ["Rest", "Stim 48hr", "stim8"],
        "donor_id": ["D1", "donor2", "3"],
        "guide_type": ["targeting", "non-targeting", "targeting"],
    })
    c = data.czi_obs_to_canonical(obs)
    assert c["pert_id"].tolist() == ["ENSG1", C.CONTROL_PERT_ID, "ENSG9"]
    assert c["condition"].tolist() == ["Rest", "Stim48hr", "Stim8hr"]
    # donor codes (real files use e.g. CE0008162) -> donor_1..N via a sorted map (4 donors)
    d4 = pd.DataFrame({
        "perturbed_gene_id": ["ENSG1"] * 4,
        "culture_condition": ["Rest"] * 4,
        "donor_id": ["CE0010866", "CE0006864", "CE0008162", "CE0008678"],
        "guide_type": ["targeting"] * 4,
    })
    dm = data.czi_donor_map(d4)
    assert dm["CE0006864"] == "donor_1" and dm["CE0010866"] == "donor_4", dm
    assert data.czi_obs_to_canonical(d4, dm)["donor"].tolist() == ["donor_4", "donor_1", "donor_2", "donor_3"]
    # a few custom spike-ins (e.g. PuroR) don't defeat the Ensembl-var detection
    assert data._frac_ensembl(["CUSTOM001_PuroR"] + [f"ENSG{i:08d}" for i in range(50)]) > 0.8
    # count normalization -> log1p CP10k, per row
    X = np.array([[1.0, 1.0, 0.0, 0.0]])
    n = data.normalize_pseudobulk_counts(X)
    assert np.isclose(n[0, 0], np.log1p(0.5 * 1e4)) and n[0, 2] == 0.0
    print("PASS  CZI obs->canonical mapping + donor map + spike-in tolerance + normalization")


def test_deg_freq_excludes_control():  # finding #7
    synthetic.write_synthetic(n_genes=120, n_perts=80, seed=1)
    deg = feat.load_deg_freq()
    assert C.CONTROL_PERT_ID not in deg.index, "control leaked into deg_freq features"
    print(f"PASS  #7 deg_freq excludes control ({len(deg)} rows, no {C.CONTROL_PERT_ID})")


def test_eval_survives_nonfinite_prediction():  # finding #6
    gt = ev.ground_truth(C.SPLIT_CONDITION)
    pred = gt.copy()
    pred.iloc[0, 0] = np.nan
    pred.iloc[1, 1] = np.inf
    m = ev.evaluate(pred, C.SPLIT_CONDITION, truth=gt)   # must NOT raise
    assert set(C.METRICS_HEADLINE) <= set(m)
    assert np.isfinite(m[C.METRIC_PEARSON_DELTA])  # other perturbations still scored
    print("PASS  #6 eval survives NaN/inf predictions (no crash, still scores good rows)")


if __name__ == "__main__":
    test_accumulator_groups_correctly()
    test_czi_obs_mapping()
    test_deg_freq_excludes_control()
    test_eval_survives_nonfinite_prediction()
    print("\nFIX REGRESSION TESTS PASSED")
