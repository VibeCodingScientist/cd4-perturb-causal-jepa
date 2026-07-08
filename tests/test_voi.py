"""Tests for ensemble-disagreement VOI + the sample-efficiency curve (§7h, S1).
Pure functions on mock prediction arrays — no trained models."""
import numpy as np
import pandas as pd
import pytest

from core.voi import (
    _first_fraction_reaching,
    ensemble_disagreement,
    gene_disagreement,
    rank_perturbations_by_voi,
    select_by_voi,
    subsampling_curve,
)


def _two_model_preds():
    genes = ["g0", "g1"]
    a = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], index=["p0", "p1"], columns=genes)
    b = pd.DataFrame([[3.0, 4.0], [1.0, 1.0]], index=["p0", "p1"], columns=genes)
    return {"A": a, "B": b}


def test_disagreement_equals_mean_pairwise_l2():
    dis = ensemble_disagreement(_two_model_preds())
    # p0: ||[0,0]-[3,4]|| = 5 ; p1: identical -> 0
    assert dis["p0"] == pytest.approx(5.0)
    assert dis["p1"] == pytest.approx(0.0)


def test_ranking_and_selection():
    dis = ensemble_disagreement(_two_model_preds())
    ranked = rank_perturbations_by_voi(dis)
    assert list(ranked.index) == ["p0", "p1"]
    assert select_by_voi(dis, 0.5) == ["p0"]


def test_disagreement_needs_two_models():
    with pytest.raises(ValueError):
        ensemble_disagreement({"only": pd.DataFrame([[1.0]], index=["p0"], columns=["g0"])})


def test_gene_disagreement_orders_genes():
    gd = gene_disagreement(_two_model_preds())
    # gene1 differs more (|-4| vs |-3| on p0) -> ranks first
    assert list(gd.index)[0] == "g1"
    assert gd["g1"] > gd["g0"]


def test_disagreement_aligns_on_shared_perturbations():
    genes = ["g0", "g1"]
    a = pd.DataFrame([[1.0, 1.0], [2.0, 2.0]], index=["p0", "p1"], columns=genes)
    b = pd.DataFrame([[1.0, 1.0], [9.0, 9.0]], index=["p0", "pX"], columns=genes)  # pX not shared
    dis = ensemble_disagreement({"A": a, "B": b})
    assert list(dis.index) == ["p0"]  # only the shared perturbation


def test_first_fraction_reaching_interpolates():
    fr = [0.25, 0.5, 1.0]
    sc = [0.2, 0.5, 1.0]
    # target 0.75 lies between (0.5, 0.5) and (1.0, 1.0) -> 0.5 + 0.5*(0.75-0.5)/(0.5) = 0.75
    assert _first_fraction_reaching(fr, sc, 0.75) == pytest.approx(0.75)
    assert _first_fraction_reaching(fr, sc, 5.0) is None  # never reached


def test_subsampling_curve_random_is_monotone_and_finds_90():
    perts = [f"p{i}" for i in range(8)]
    fractions = [0.25, 0.5, 1.0]

    def train_eval(selected, seed):  # score grows with how many perturbations we measure
        return len(set(selected)) / len(perts)

    curve = subsampling_curve(perts, fractions, train_eval, voi_scores=None, seed=0)
    assert curve.random_mean == sorted(curve.random_mean)  # non-decreasing
    assert curve.full_score == pytest.approx(1.0)
    assert curve.random_90_fraction == pytest.approx(0.9, abs=1e-6)


def test_voi_guided_reaches_target_sooner_than_random():
    perts = [f"p{i}" for i in range(8)]
    voi = pd.Series([8, 7, 6, 5, 4, 3, 2, 1], index=perts, dtype=float)
    informative = set(voi.sort_values(ascending=False).index[:4])  # top-4 by VOI

    def train_eval(selected, seed):
        return min(1.0, len(set(selected) & informative) / 4.0)

    curve = subsampling_curve(perts, [0.25, 0.5, 1.0], train_eval, voi_scores=voi, seed=0)
    # VOI-guided selects the informative perturbations first -> hits 90% at a small fraction
    assert curve.voi_90_fraction is not None
    assert curve.voi_90_fraction < (curve.random_90_fraction or 1.0)
    assert curve.voi_score[curve.fractions.index(0.5)] >= 0.9
