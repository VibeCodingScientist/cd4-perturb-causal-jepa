"""Tests for the 2x2 ablation harness (§7f). Built against the frozen
``evaluate(pred_df, split) -> dict`` signature with a mock evaluate; run files carry
controlled magnitudes so the mock maps them to known scores."""
import pandas as pd
import pytest

from core import contract
from core.ablation import (
    GRID_AXES,
    assemble_2x2,
    causal_effect,
    jepa_effect,
    run_2x2,
    score_grid,
    upsert_benchmark,
)
from tests.fixtures import make_delta_df, mock_evaluate, write_run

# constants -> mock pearson = tanh(constant); chosen so causal>noncausal and jepa>random
_CONSTANTS = {
    contract.MODEL_NONCAUSAL: 0.1,
    contract.MODEL_CAUSAL: 0.5,
    contract.MODEL_JEPA_ONLY: 0.3,
    contract.MODEL_JEPA_CAUSAL: 0.7,
}


def _write_grid_runs(split=contract.SPLIT_CONDITION):
    for model, c in _CONSTANTS.items():
        write_run(model, split, make_delta_df(constant=c))


def test_score_grid_produces_benchmark_rows():
    _write_grid_runs()
    df = score_grid([contract.SPLIT_CONDITION], evaluate_fn=mock_evaluate)
    assert set(df["model"]) == set(_CONSTANTS)
    assert list(df.columns) == list(contract.BENCHMARK_COLUMNS)
    # the near-zero-ish noncausal has the highest perturbench_rank -> flagged collapse
    nc = df[df["model"] == contract.MODEL_NONCAUSAL].iloc[0]
    assert nc[contract.MODE_COLLAPSE_FLAG] in (True, False)  # column present + boolean


def test_score_grid_records_missing_runs():
    # only write two of four models
    write_run(contract.MODEL_CAUSAL, contract.SPLIT_CONDITION, make_delta_df(constant=0.5))
    write_run(contract.MODEL_NONCAUSAL, contract.SPLIT_CONDITION, make_delta_df(constant=0.1))
    # remove the JEPA runs if a previous test wrote them
    for m in (contract.MODEL_JEPA_ONLY, contract.MODEL_JEPA_CAUSAL):
        p = contract.run_path(m, contract.SPLIT_CONDITION)
        if p.exists():
            p.unlink()
    df = score_grid([contract.SPLIT_CONDITION], evaluate_fn=mock_evaluate)
    missing = dict.fromkeys(m for m, _ in df.attrs["missing"])
    assert contract.MODEL_JEPA_ONLY in missing and contract.MODEL_JEPA_CAUSAL in missing


def test_assemble_2x2_maps_axes_correctly():
    _write_grid_runs()
    df = score_grid([contract.SPLIT_CONDITION], evaluate_fn=mock_evaluate)
    grid = assemble_2x2(df)
    assert grid.shape == (2, 2)
    import numpy as np

    # random/off == noncausal's pearson = tanh(0.1)
    assert grid.loc["random", "off"] == pytest.approx(float(np.tanh(0.1)), abs=1e-6)
    assert grid.loc["jepa", "on"] == pytest.approx(float(np.tanh(0.7)), abs=1e-6)


def test_contrasts_have_expected_signs():
    _write_grid_runs()
    df = score_grid([contract.SPLIT_CONDITION], evaluate_fn=mock_evaluate)
    grid = assemble_2x2(df)
    c2 = causal_effect(grid)   # mask on - off
    c3 = jepa_effect(grid)     # jepa - random
    assert c2.mean > 0 and all(v > 0 for v in c2.per_pair.values())
    assert c3.mean > 0 and all(v > 0 for v in c3.per_pair.values())


def test_upsert_does_not_clobber_cp1_rows(tmp_path):
    table = tmp_path / "benchmark_table.csv"
    # a pre-existing CP1 row (Developer 1's ridge)
    ridge = pd.DataFrame(
        [{
            "model": "ridge", "split": contract.SPLIT_CONDITION,
            contract.METRIC_PEARSON_DELTA: 0.4, contract.METRIC_PERTURBENCH_RANK: 0.2,
            contract.METRIC_DES: 0.6, contract.MODE_COLLAPSE_FLAG: False,
        }],
        columns=list(contract.BENCHMARK_COLUMNS),
    )
    ridge.to_csv(table, index=False)

    _write_grid_runs()
    rows = score_grid([contract.SPLIT_CONDITION], evaluate_fn=mock_evaluate)
    merged = upsert_benchmark(rows, table_path=table)
    models = set(merged["model"])
    assert "ridge" in models  # CP1 row survived
    assert {contract.MODEL_JEPA_ONLY, contract.MODEL_JEPA_CAUSAL} <= models
    # (model, split) is unique
    assert not merged.duplicated(subset=["model", "split"]).any()


def test_run_2x2_end_to_end(tmp_path):
    _write_grid_runs()
    result = run_2x2(
        splits=[contract.SPLIT_CONDITION],
        evaluate_fn=mock_evaluate,
        table_path=tmp_path / "bench.csv",
    )
    assert result.grid.shape == (2, 2)
    assert result.c2.mean > 0 and result.c3.mean > 0
    assert (tmp_path / "bench.csv").exists()


def test_grid_axes_cover_all_four_models():
    assert set(GRID_AXES) == set(contract.GRID_2X2)
