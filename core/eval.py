"""
core.eval — FROZEN evaluation harness (UNIFIED_BUILD_PLAN.md §5, §7i).

Every model is scored by the SAME `evaluate(pred_delta_df, split)` and its result row
is appended to `results/benchmark_table.csv`. No model reimplements a metric.

Frozen public signature (declared in core.contract):

    evaluate(pred_delta_df: pd.DataFrame, split: str) -> dict[str, float | bool]

  * `pred_delta_df`: index = pert_id, columns = HVG gene ids, values = predicted
    DELTA (post - control). Exactly the schema written to `contract.run_path(...)`.
  * `split`: one of contract.ALL_SPLITS; selects the frozen ground-truth test set.
  * returns: dict keyed by the metric-name constants in core.contract, plus the
    boolean mode-collapse flag. The caller fills in "model"/"split" before appending.

Ground truth. In production the per-perturbation true delta is derived from the frozen
`pseudobulk/test.parquet` (delta block) via `ground_truth(split)`. For unit tests, pass
a `truth=` DataFrame (index = pert_id, columns = HVG) to bypass disk — the frozen
signature is preserved because `truth` is keyword-only.

Metric conventions
------------------
* pearson_delta_top50  (HEADLINE, higher better): per-perturbation Pearson between
  predicted and true delta over that perturbation's top-50 genes by |true delta|,
  averaged across perturbations.
* perturbench_rank     (HEADLINE, LOWER better; >0.4 = red): perturbation-discrimination
  / mode-collapse detector. For each perturbation, the normalized rank of its own true
  delta among all true deltas by distance to the prediction. A model that predicts one
  profile for everything scores ~0.5 (random) and is flagged.
* des                  (HEADLINE, higher better): sign-correct DEG overlap — fraction of
  each perturbation's top-50 DEGs whose predicted delta sign matches the true sign.
* mae, spearman_lfc, spearman_effect, auprc, edistance: full appendix battery.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from . import contract as C

# How many top-DEGs the headline metrics restrict to.
_TOP_K = C.TOP_DEG_N  # 50


# ---------------------------------------------------------------------------
# Ground-truth assembly
# ---------------------------------------------------------------------------
def ground_truth(split: str, *, pseudobulk_test: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-perturbation true delta for a split (index = pert_id, columns = HVG).

    Loads the frozen `pseudobulk/test.parquet` delta block (or an injected frame) and
    reduces the (pert_id, condition, donor) rows to one true delta per perturbation by
    averaging over the nuisance axes appropriate to each split (§3):

      * condition : rows at the held-out condition (Stim48hr), averaged over donor.
      * gene      : rows for held-out genes at the TRAIN conditions, averaged over
                    condition + donor (pure interpolation over unseen genes).
      * donor     : rows for the donor probe, averaged over condition.

    The non-targeting control is dropped (its delta is ~0 by construction).
    """
    if split not in C.ALL_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {C.ALL_SPLITS}")
    if pseudobulk_test is None:
        pseudobulk_test = pd.read_parquet(C.PSEUDOBULK_TEST)
    delta = C.pseudobulk_delta(pseudobulk_test)  # (pert,cond,donor) x genes

    idx = delta.index
    pert = idx.get_level_values("pert_id")
    cond = idx.get_level_values("condition")

    if split == C.SPLIT_CONDITION:
        mask = (cond == C.CONDITION_HOLDOUT) & (pert != C.CONTROL_PERT_ID)
    elif split == C.SPLIT_GENE:
        man = load_manifest()
        held = set(man.gene_holdout)
        mask = np.isin(pert, list(held)) & np.isin(cond, list(C.TRAIN_CONDITIONS))
    else:  # SPLIT_DONOR
        donor = idx.get_level_values("donor")
        mask = (donor == C.DONOR_PROBE) & (pert != C.CONTROL_PERT_ID)

    sub = delta[mask]
    if sub.empty:
        raise ValueError(f"no ground-truth rows for split {split!r}")
    # collapse to one row per perturbation
    return sub.groupby(level="pert_id").mean()


def load_manifest() -> "C.SplitManifest":
    import json
    return C.SplitManifest.from_dict(json.loads(C.SPLIT_MANIFEST.read_text()))


def evaluable_perts(split: str, *, pseudobulk_test: Optional[pd.DataFrame] = None) -> list[str]:
    """The canonical perturbation set every model predicts for a split, so the benchmark is
    apples-to-apples: the split's test perturbations whose silenced gene is in the HVG panel.

    The causal transformer represents the perturbed gene as a token in the sequence, so it is
    intrinsically scoped to HVG-panel perturbations; scoring the baselines on the same set
    keeps every benchmark row comparable. Falls back to all test perturbations if the HVG list
    is unavailable (e.g. a bare synthetic run without a frozen split).
    """
    perts = list(ground_truth(split, pseudobulk_test=pseudobulk_test).index)
    try:
        from . import split as _split
        panel = set(_split.load_hvg())
    except Exception:
        return perts
    filtered = [p for p in perts if p in panel]
    return filtered or perts


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _align(pred: pd.DataFrame, truth: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Intersect perturbations (rows) and genes (columns); return aligned arrays."""
    perts = [p for p in truth.index if p in pred.index]
    if not perts:
        raise ValueError("no overlapping perturbations between prediction and ground truth")
    genes = [g for g in truth.columns if g in pred.columns]
    if not genes:
        raise ValueError("no overlapping genes between prediction and ground truth")
    P = pred.loc[perts, genes].to_numpy(dtype=float)
    T = truth.loc[perts, genes].to_numpy(dtype=float)
    return P, T, perts


def _topk_idx(true_row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k genes with the largest |true delta| for one perturbation."""
    k = min(k, true_row.size)
    return np.argpartition(np.abs(true_row), -k)[-k:]


# ---------------------------------------------------------------------------
# Individual metrics (each takes aligned P, T arrays of shape (n_pert, n_gene))
# ---------------------------------------------------------------------------
def _pearson_delta_topk(P: np.ndarray, T: np.ndarray, k: int) -> float:
    vals = []
    for i in range(P.shape[0]):
        idx = _topk_idx(T[i], k)
        pt, tt = P[i, idx], T[i, idx]
        if np.std(pt) < 1e-12 or np.std(tt) < 1e-12:
            continue  # undefined correlation (constant vector)
        vals.append(pearsonr(pt, tt)[0])
    return float(np.nanmean(vals)) if vals else float("nan")


def _des(P: np.ndarray, T: np.ndarray, k: int) -> float:
    """Sign-correct DEG overlap over each perturbation's top-k DEGs."""
    vals = []
    for i in range(P.shape[0]):
        idx = _topk_idx(T[i], k)
        sp, st = np.sign(P[i, idx]), np.sign(T[i, idx])
        nz = st != 0
        if nz.sum() == 0:
            continue
        vals.append(float(np.mean(sp[nz] == st[nz])))
    return float(np.mean(vals)) if vals else float("nan")


def _perturbench_rank(P: np.ndarray, T: np.ndarray) -> float:
    """Perturbation-discrimination / mode-collapse detector (LOWER better).

    For each perturbation i, distance from prediction P_i to every true profile T_j;
    normalized rank of the correct profile T_i = (#{j!=i : d_ij < d_ii}) / (n-1). A
    model that predicts (nearly) the same delta for all perturbations pushes every
    correct profile to a random rank ~0.5. Uses Euclidean distance in delta space.
    """
    n = P.shape[0]
    if n < 2:
        return float("nan")
    # d[i,j] = || P_i - T_j ||_2
    # (P_i·P_i) - 2 P_i·T_j + (T_j·T_j)
    PP = np.einsum("ij,ij->i", P, P)[:, None]
    TT = np.einsum("ij,ij->i", T, T)[None, :]
    d2 = PP - 2.0 * (P @ T.T) + TT
    d2 = np.maximum(d2, 0.0)
    diag = np.diag(d2)[:, None]
    # strictly-closer competitors; the diagonal d_ii<d_ii is False so it never counts.
    closer = (d2 < diag).sum(axis=1)
    # ties (d_ij == d_ii) include the self term j==i, so subtract 1; count them at half.
    ties = (d2 == diag).sum(axis=1) - 1
    rank = (closer + 0.5 * ties) / (n - 1)
    return float(np.mean(rank))


def _mae(P: np.ndarray, T: np.ndarray) -> float:
    return float(np.mean(np.abs(P - T)))


def _spearman_lfc(P: np.ndarray, T: np.ndarray) -> float:
    vals = []
    for i in range(P.shape[0]):
        if np.std(P[i]) < 1e-12 or np.std(T[i]) < 1e-12:
            continue
        vals.append(spearmanr(P[i], T[i]).correlation)
    return float(np.nanmean(vals)) if vals else float("nan")


def _spearman_effect(P: np.ndarray, T: np.ndarray) -> float:
    """Spearman over effect magnitudes |delta| per perturbation, averaged."""
    vals = []
    for i in range(P.shape[0]):
        p, t = np.abs(P[i]), np.abs(T[i])
        if np.std(p) < 1e-12 or np.std(t) < 1e-12:
            continue
        vals.append(spearmanr(p, t).correlation)
    return float(np.nanmean(vals)) if vals else float("nan")


def _auprc(P: np.ndarray, T: np.ndarray, k: int) -> float:
    """DE-detection AUPRC: label = gene in the perturbation's top-k true DEGs,
    score = |predicted delta|. Averaged across perturbations."""
    vals = []
    n_gene = T.shape[1]
    for i in range(P.shape[0]):
        idx = _topk_idx(T[i], k)
        y = np.zeros(n_gene, dtype=int)
        y[idx] = 1
        if y.sum() == 0 or y.sum() == n_gene:
            continue
        vals.append(average_precision_score(y, np.abs(P[i])))
    return float(np.mean(vals)) if vals else float("nan")


def _edistance(P: np.ndarray, T: np.ndarray) -> float:
    """Energy distance between the predicted-delta and true-delta point clouds
    (across perturbations). LOWER better; 0 iff the two clouds coincide."""
    def _mean_pairwise(A, B):
        # mean Euclidean distance between rows of A and rows of B
        AA = np.einsum("ij,ij->i", A, A)[:, None]
        BB = np.einsum("ij,ij->i", B, B)[None, :]
        d2 = np.maximum(AA - 2.0 * (A @ B.T) + BB, 0.0)
        return float(np.mean(np.sqrt(d2)))
    d_pt = _mean_pairwise(P, T)
    d_pp = _mean_pairwise(P, P)
    d_tt = _mean_pairwise(T, T)
    return float(max(2.0 * d_pt - d_pp - d_tt, 0.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate(
    pred_delta_df: pd.DataFrame,
    split: str,
    *,
    truth: Optional[pd.DataFrame] = None,
    full: bool = True,
) -> Dict[str, float | bool]:
    """Score a prediction against the frozen ground truth for `split`.

    Returns a dict with the headline metrics always, the full battery when `full`,
    and the boolean mode-collapse flag. `truth` (keyword-only) overrides the on-disk
    ground truth for testing; the frozen public signature is `evaluate(df, split)`.
    """
    if truth is None:
        truth = ground_truth(split)
    P, T, _ = _align(pred_delta_df, truth)

    out: Dict[str, float | bool] = {
        C.METRIC_PEARSON_DELTA: _pearson_delta_topk(P, T, _TOP_K),
        C.METRIC_PERTURBENCH_RANK: _perturbench_rank(P, T),
        C.METRIC_DES: _des(P, T, _TOP_K),
    }
    if full:
        out.update({
            C.METRIC_MAE: _mae(P, T),
            C.METRIC_SPEARMAN_LFC: _spearman_lfc(P, T),
            C.METRIC_SPEARMAN_EFFECT: _spearman_effect(P, T),
            C.METRIC_AUPRC: _auprc(P, T, _TOP_K),
            C.METRIC_EDISTANCE: _edistance(P, T),
        })
    out[C.MODE_COLLAPSE_FLAG] = bool(
        out[C.METRIC_PERTURBENCH_RANK] > C.MODE_COLLAPSE_THRESHOLD
    )
    return out


def evaluate_and_record(
    pred_delta_df: pd.DataFrame,
    split: str,
    model_name: str,
    *,
    truth: Optional[pd.DataFrame] = None,
) -> Dict[str, float | bool]:
    """Evaluate then append a row to both benchmark tables (headline + full)."""
    if model_name not in C.MODELS:
        raise ValueError(f"unknown model {model_name!r}; expected one of {C.MODELS}")
    metrics = evaluate(pred_delta_df, split, truth=truth, full=True)
    row = {"model": model_name, "split": split, **metrics}
    _append_row(C.BENCHMARK_TABLE, row, list(C.BENCHMARK_COLUMNS))
    _append_row(C.BENCHMARK_TABLE_FULL, row, list(C.BENCHMARK_COLUMNS_FULL))
    return metrics


def score_run_file(model_name: str, split: str) -> Dict[str, float | bool]:
    """Score a model's `runs/<model>_<split>.parquet` and record it (Lane C entrypoint)."""
    pred = pd.read_parquet(C.run_path(model_name, split))
    return evaluate_and_record(pred, split, model_name)


def _append_row(path, row: dict, columns: list[str]) -> None:
    """Idempotent upsert on (model, split): replace an existing row, else append."""
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([{c: row.get(c) for c in columns}], columns=columns)
    if path.exists():
        old = pd.read_csv(path)
        # drop any prior row for this (model, split) so re-runs overwrite cleanly
        keep = ~((old["model"] == row["model"]) & (old["split"] == row["split"]))
        out = pd.concat([old[keep], new], ignore_index=True)
    else:
        out = new
    out = out.sort_values(["split", "model"]).reset_index(drop=True)
    out.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Score a model run and append to the benchmark table.")
    ap.add_argument("model", choices=list(C.MODELS))
    ap.add_argument("split", choices=list(C.ALL_SPLITS))
    args = ap.parse_args()
    m = score_run_file(args.model, args.split)
    print(f"{args.model} / {args.split}:")
    for k, v in m.items():
        print(f"  {k:22s} {v}")
