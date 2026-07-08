#!/usr/bin/env python
"""CP2 finalization (CPU, runs after G5): assemble the 2x2, compute VOI + the
sample-efficiency curve, and render the demo figures. No GPU needed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import contract as C
from core import eval as ev
from core.ablation import assemble_2x2, causal_effect, jepa_effect, upsert_benchmark, score_grid
from core import voi as voi_mod
from figures import make_figures as F

CONDITION = C.SPLIT_CONDITION


def load_benchmark() -> pd.DataFrame:
    if C.BENCHMARK_TABLE.exists():
        return pd.read_csv(C.BENCHMARK_TABLE)
    # fall back: score whatever run files exist via the real evaluate
    return score_grid(C.SPLITS)


def ensemble_preds(split: str, models) -> dict:
    out = {}
    for m in models:
        p = C.run_path(m, split)
        if p.exists():
            out[m] = pd.read_parquet(p)
    return out


def ridge_subsampling(fractions, voi_scores, n_random=3, seed=42):
    """Sample-efficiency curve using Ridge (fast, CPU) retrained on perturbation subsets,
    evaluated on the condition hold-out (§7h). VOI-guided selection uses the ensemble
    disagreement scores."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from core.models.baselines import train_pert_targets, split_test_perts
    from core import features as feat

    y = train_pert_targets()
    X = feat.pert_gene_features(y.index)
    genes = list(y.columns)
    test_perts = split_test_perts(CONDITION)
    Xte = feat.pert_gene_features(test_perts)

    def train_eval(subset, s):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(X.loc[subset].to_numpy(), y.loc[subset].to_numpy())
        pred = pd.DataFrame(model.predict(Xte.to_numpy()), index=test_perts, columns=genes)
        pred.index.name = "pert_id"
        return ev.evaluate(pred, CONDITION)[C.METRIC_PEARSON_DELTA]

    # VOI scores are per-test-pert; map to training perts we can actually subsample.
    train_perts = list(y.index)
    vs = voi_scores.reindex(train_perts).dropna() if voi_scores is not None else None
    return voi_mod.subsampling_curve(train_perts, fractions, train_eval, voi_scores=vs,
                                     n_random_replicates=n_random, seed=seed)


def main():
    C.ensure_dirs()
    bench = load_benchmark()
    print("=== benchmark rows ===")
    print(bench[["model", "split", C.METRIC_PEARSON_DELTA, C.METRIC_PERTURBENCH_RANK,
                 C.MODE_COLLAPSE_FLAG]].to_string(index=False))

    # ---- 2x2 + contrasts (C2 do-operator, C3 JEPA-init) ------------------
    grid = assemble_2x2(bench, split=CONDITION)
    c2, c3 = causal_effect(grid), jepa_effect(grid)
    print("\n=== 2x2 (condition hold-out Pearson-delta) ===")
    print(grid.to_string())
    print(f"\n{c2}\n{c3}")

    # ---- VOI disagreement ------------------------------------------------
    models = [C.MODEL_RIDGE, C.MODEL_CAUSAL, C.MODEL_NONCAUSAL,
              C.MODEL_JEPA_CAUSAL, C.MODEL_JEPA_ONLY]
    preds = ensemble_preds(CONDITION, models)
    curve = None
    gene_dis = None
    if len(preds) >= 2:
        pert_dis = voi_mod.ensemble_disagreement(preds)
        gene_dis = voi_mod.gene_disagreement(preds)
        print(f"\n=== VOI: {len(preds)} models; top-5 most-worth-measuring perturbations ===")
        print(voi_mod.rank_perturbations_by_voi(pert_dis).head().to_string())
        # ---- sample-efficiency curve -------------------------------------
        try:
            curve = ridge_subsampling([0.05, 0.1, 0.2, 0.5, 1.0], pert_dis)
            print(f"\n=== sample-efficiency: full={curve.full_score:.3f}; "
                  f"90% reached at random={curve.random_90_fraction}, voi={curve.voi_90_fraction}")
        except Exception as e:
            print(f"[cp2] subsampling curve skipped: {e}")
    else:
        print("[cp2] <2 run files for condition split; VOI skipped")

    # ---- figures ---------------------------------------------------------
    paths = {}
    paths["figure1"] = F.figure1_benchmark(bench, out_path=C.FIGURES_DIR / "figure1_benchmark.png")
    paths["figure2"] = F.figure2_2x2(grid, out_path=C.FIGURES_DIR / "figure2_2x2.png", c2=c2, c3=c3)
    if curve is not None:
        paths["figure3"] = F.figure3_subsampling(curve, out_path=C.FIGURES_DIR / "figure3_subsampling.png")
    if gene_dis is not None:
        bio = F.build_biology_annotation(gene_dis, top_n=20)
        paths["figure4"] = F.figure4_biology(bio, out_path=C.FIGURES_DIR / "figure4_biology.png")
    print("\n=== figures ===")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("\nCP2 finalize done.")


if __name__ == "__main__":
    main()
