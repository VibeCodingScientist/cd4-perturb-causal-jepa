"""Tests for the demo figures (§12). Render from mock artifacts into tmp paths so the
committed figures/ dir is never touched; assert each PNG is produced and non-empty."""
import numpy as np
import pandas as pd

from core import contract
from figures.make_figures import (
    build_biology_annotation,
    figure1_benchmark,
    figure2_2x2,
    figure3_subsampling,
    figure4_biology,
    make_all_figures,
)


def _mock_benchmark():
    rows = []
    for model, pear, rank in [
        ("ridge", 0.40, 0.20),
        ("causal", 0.55, 0.15),
        ("noncausal", 0.02, 0.48),   # mode-collapsed -> red
        ("jepa_causal", 0.60, 0.12),
    ]:
        rows.append({
            "model": model, "split": contract.SPLIT_CONDITION,
            contract.METRIC_PEARSON_DELTA: pear,
            contract.METRIC_PERTURBENCH_RANK: rank,
            contract.METRIC_DES: 0.5 + pear / 2,
            contract.MODE_COLLAPSE_FLAG: rank > contract.MODE_COLLAPSE_THRESHOLD,
        })
    return pd.DataFrame(rows, columns=list(contract.BENCHMARK_COLUMNS))


def _assert_png(path):
    from pathlib import Path

    p = Path(path)
    assert p.exists() and p.stat().st_size > 0


def test_figure1_benchmark(tmp_path):
    out = figure1_benchmark(_mock_benchmark(), out_path=tmp_path / "f1.png")
    _assert_png(out)


def test_figure2_2x2(tmp_path):
    grid = pd.DataFrame(
        [[0.10, 0.30], [0.25, 0.45]],
        index=pd.Index(["random", "jepa"], name="init"),
        columns=pd.Index(["off", "on"], name="mask"),
    )
    from core.ablation import causal_effect, jepa_effect

    out = figure2_2x2(grid, out_path=tmp_path / "f2.png", c2=causal_effect(grid), c3=jepa_effect(grid))
    _assert_png(out)


def test_figure3_subsampling(tmp_path):
    from core.voi import subsampling_curve

    perts = [f"p{i}" for i in range(8)]
    voi = pd.Series(np.arange(8, 0, -1), index=perts, dtype=float)
    curve = subsampling_curve(perts, [0.25, 0.5, 1.0], lambda s, _seed: len(set(s)) / 8, voi_scores=voi, seed=0)
    out = figure3_subsampling(curve, out_path=tmp_path / "f3.png")
    _assert_png(out)


def test_figure4_biology(tmp_path):
    gd = pd.Series([3.0, 2.0, 1.0], index=["IL2RA", "FOXP3", "CTLA4"], name="gene_disagreement")
    ann = pd.DataFrame(
        {"gene_family": ["receptor", "TF", "receptor"],
         "tcell_role": ["activation", "Treg", "inhibitory"],
         "gwas_flag": [True, False, True]},
        index=["IL2RA", "FOXP3", "CTLA4"],
    )
    bio = build_biology_annotation(gd, ann, top_n=3)
    assert list(bio["gene"]) == ["IL2RA", "FOXP3", "CTLA4"]
    assert bool(bio.loc[bio["gene"] == "IL2RA", "gwas_flag"].iloc[0]) is True
    out = figure4_biology(bio, out_path=tmp_path / "f4.png")
    _assert_png(out)


def test_build_biology_annotation_without_annotations():
    gd = pd.Series([2.0, 1.0], index=["A", "B"])
    bio = build_biology_annotation(gd, annotations=None, top_n=2)
    assert (bio["gene_family"] == "unknown").all()
    assert (~bio["gwas_flag"]).all()


def test_make_all_figures_returns_paths(tmp_path, monkeypatch):
    # redirect FIGURES_DIR into tmp so defaults don't touch the repo
    monkeypatch.setattr(contract, "FIGURES_DIR", tmp_path)
    import figures.make_figures as mf

    monkeypatch.setattr(mf.contract, "FIGURES_DIR", tmp_path)
    grid = pd.DataFrame(
        [[0.1, 0.3], [0.25, 0.45]],
        index=pd.Index(["random", "jepa"], name="init"),
        columns=pd.Index(["off", "on"], name="mask"),
    )
    paths = make_all_figures(benchmark=_mock_benchmark(), grid=grid)
    assert "figure1" in paths and "figure2" in paths
    for p in paths.values():
        _assert_png(p)
