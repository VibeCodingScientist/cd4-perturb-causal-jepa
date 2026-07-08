"""core.ablation — the 2x2 experimental core (§7f) and its contrasts.

The 2x2 is encoder-init x causal-mask:

               mask OFF          mask ON
    random     noncausal         causal        (Developer 1's G2/G3 runs)
    JEPA       jepa_only         jepa_causal   (Developer 2's G5 runs)

This module *scores* the four models on the frozen splits via the contract's
``core.eval.evaluate(pred_delta_df, split) -> dict`` signature, upserts their rows
into ``results/benchmark_table.csv`` (without clobbering the CP1 rows Developer 1
already wrote), assembles the 2x2 matrix on condition-hold-out Pearson-delta, and
reads out the two pre-registered contrasts:

    C2 (do-operator) = causal - noncausal   and   jepa_causal - jepa_only
    C3 (JEPA-init)   = jepa_only - noncausal and   jepa_causal - causal

Developer 2 does NOT retrain the random-init cells — they come straight from
Developer 1's ``runs/*.parquet``. Built against the ``evaluate`` signature with an
injectable ``evaluate_fn`` so it tests on mock metric dicts (no trained models).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import pandas as pd

from core import contract

EvaluateFn = Callable[[pd.DataFrame, str], dict]

# model -> (encoder init, causal mask) — the 2x2 axes.
GRID_AXES: Dict[str, tuple[str, str]] = {
    contract.MODEL_NONCAUSAL: ("random", "off"),
    contract.MODEL_CAUSAL: ("random", "on"),
    contract.MODEL_JEPA_ONLY: ("jepa", "off"),
    contract.MODEL_JEPA_CAUSAL: ("jepa", "on"),
}


def _default_evaluate() -> EvaluateFn:
    from core.eval import evaluate  # Developer 1, appears at core-frozen
    return evaluate


# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------
def _benchmark_row(model_name: str, split: str, metrics: dict, columns: Sequence[str]) -> dict:
    """Assemble a benchmark-table row from an ``evaluate`` metric dict, filling the
    mode-collapse flag from perturbench_rank (§7i)."""
    row = {"model": model_name, "split": split}
    for col in columns:
        if col in (contract.BENCHMARK_ID_COLUMNS + (contract.MODE_COLLAPSE_FLAG,)):
            continue
        row[col] = metrics.get(col)
    rank = metrics.get(contract.METRIC_PERTURBENCH_RANK)
    row[contract.MODE_COLLAPSE_FLAG] = bool(rank is not None and rank > contract.MODE_COLLAPSE_THRESHOLD)
    return row


def score_model(
    model_name: str,
    split: str,
    evaluate_fn: Optional[EvaluateFn] = None,
    full: bool = False,
) -> Optional[dict]:
    """Score one model on one split from its ``runs/<model>_<split>.parquet``.

    Returns a benchmark-table row dict, or None if the run file is absent.
    """
    path = contract.run_path(model_name, split)
    if not path.exists():
        return None
    pred_delta_df = pd.read_parquet(path)
    evaluate_fn = evaluate_fn or _default_evaluate()
    metrics = evaluate_fn(pred_delta_df, split)
    columns = contract.BENCHMARK_COLUMNS_FULL if full else contract.BENCHMARK_COLUMNS
    return _benchmark_row(model_name, split, metrics, columns)


def score_grid(
    splits: Sequence[str] = contract.SPLITS,
    evaluate_fn: Optional[EvaluateFn] = None,
    models: Sequence[str] = contract.GRID_2X2,
    full: bool = False,
) -> pd.DataFrame:
    """Score every 2x2 model on every split. Missing runs are skipped (with a note in
    the returned frame's attrs)."""
    rows, missing = [], []
    for model in models:
        for split in splits:
            row = score_model(model, split, evaluate_fn=evaluate_fn, full=full)
            if row is None:
                missing.append((model, split))
            else:
                rows.append(row)
    columns = contract.BENCHMARK_COLUMNS_FULL if full else contract.BENCHMARK_COLUMNS
    df = pd.DataFrame(rows, columns=list(columns))
    df.attrs["missing"] = missing
    return df


# ---------------------------------------------------------------------------
# 2. Benchmark table upsert (don't clobber CP1 rows)
# ---------------------------------------------------------------------------
def upsert_benchmark(
    rows: pd.DataFrame,
    table_path=None,
    columns: Sequence[str] = contract.BENCHMARK_COLUMNS,
) -> pd.DataFrame:
    """Merge ``rows`` into the benchmark CSV, replacing any existing (model, split)
    pairs and appending the rest. Returns the merged table (also written to disk)."""
    table_path = Path(table_path) if table_path is not None else contract.BENCHMARK_TABLE
    columns = list(columns)
    if table_path.exists():
        existing = pd.read_csv(table_path)
    else:
        existing = None
        table_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = rows[columns] if len(rows) else pd.DataFrame(columns=columns)
    frames = [f for f in (existing, new_rows) if f is not None and len(f)]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    # keep the LAST occurrence of each (model, split) so new rows win
    combined = combined.drop_duplicates(subset=list(contract.BENCHMARK_ID_COLUMNS), keep="last")
    combined = combined.sort_values(list(contract.BENCHMARK_ID_COLUMNS)).reset_index(drop=True)
    combined.to_csv(table_path, index=False)
    return combined


# ---------------------------------------------------------------------------
# 3. The 2x2 matrix + contrasts
# ---------------------------------------------------------------------------
@dataclass
class Contrast:
    name: str
    per_pair: Dict[str, float]        # e.g. {"random": causal-noncausal, "jepa": jepa_causal-jepa_only}
    mean: float

    def __str__(self):
        pairs = ", ".join(f"{k}: {v:+.4f}" for k, v in self.per_pair.items())
        return f"{self.name}: mean {self.mean:+.4f} ({pairs})"


def assemble_2x2(
    benchmark: pd.DataFrame,
    split: str = contract.SPLIT_CONDITION,
    metric: str = contract.METRIC_PEARSON_DELTA,
) -> pd.DataFrame:
    """2x2 matrix (rows: init random/jepa; cols: mask off/on) of ``metric`` on ``split``.

    NaN where the corresponding run is missing.
    """
    sub = benchmark[benchmark["split"] == split].set_index("model")
    grid = pd.DataFrame(
        index=pd.Index(["random", "jepa"], name="init"),
        columns=pd.Index(["off", "on"], name="mask"),
        dtype=float,
    )
    for model, (init, mask) in GRID_AXES.items():
        grid.loc[init, mask] = sub[metric].get(model, float("nan")) if model in sub.index else float("nan")
    return grid


def causal_effect(grid: pd.DataFrame) -> Contrast:
    """C2: do-operator isolation = (mask on - mask off), per init row."""
    per = {init: float(grid.loc[init, "on"] - grid.loc[init, "off"]) for init in grid.index}
    finite = [v for v in per.values() if pd.notna(v)]
    return Contrast("C2 (do-operator: mask on - off)", per, float(sum(finite) / len(finite)) if finite else float("nan"))


def jepa_effect(grid: pd.DataFrame) -> Contrast:
    """C3: JEPA-init effect = (jepa init - random init), per mask column."""
    per = {mask: float(grid.loc["jepa", mask] - grid.loc["random", mask]) for mask in grid.columns}
    finite = [v for v in per.values() if pd.notna(v)]
    return Contrast("C3 (JEPA-init: jepa - random)", per, float(sum(finite) / len(finite)) if finite else float("nan"))


@dataclass
class AblationResult:
    benchmark: pd.DataFrame
    grid: pd.DataFrame
    c2: Contrast
    c3: Contrast
    split: str
    metric: str


def run_2x2(
    splits: Sequence[str] = contract.SPLITS,
    evaluate_fn: Optional[EvaluateFn] = None,
    write: bool = True,
    metric: str = contract.METRIC_PEARSON_DELTA,
    table_path=None,
) -> AblationResult:
    """Score the four models, upsert the benchmark table, assemble the condition
    hold-out 2x2, and compute the C2 / C3 contrasts. The one call for CP2."""
    # A full-battery metric (e.g. mae) lives only in the FULL columns, so score/upsert
    # against the matching column set or assemble_2x2 would KeyError on it.
    full = metric not in contract.METRICS_HEADLINE
    columns = contract.BENCHMARK_COLUMNS_FULL if full else contract.BENCHMARK_COLUMNS
    scored = score_grid(splits, evaluate_fn=evaluate_fn, full=full)
    if write and len(scored):
        upsert_benchmark(scored, table_path=table_path, columns=columns)
    grid = assemble_2x2(scored, split=contract.SPLIT_CONDITION, metric=metric)
    return AblationResult(
        benchmark=scored,
        grid=grid,
        c2=causal_effect(grid),
        c3=jepa_effect(grid),
        split=contract.SPLIT_CONDITION,
        metric=metric,
    )


__all__ = [
    "GRID_AXES", "score_model", "score_grid", "upsert_benchmark",
    "assemble_2x2", "causal_effect", "jepa_effect",
    "Contrast", "AblationResult", "run_2x2",
]
