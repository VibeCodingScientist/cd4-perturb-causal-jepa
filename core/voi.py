"""core.voi — Value-of-Information from ensemble disagreement (§7h, claim S1).

The applied layer: if the model ensemble *disagrees* about a perturbation's effect,
that perturbation carries high epistemic uncertainty and is the most worth measuring
next. No normalizing-flow dependency (§7h) — VOI is the **mean pairwise L2
disagreement** across the ensemble's per-gene delta predictions.

Two public surfaces:
  * ``ensemble_disagreement`` / ``rank_perturbations_by_voi`` — the score + ranking.
  * ``subsampling_curve`` — train the ensemble on 5/10/20/50/100% of training
    perturbations (3 random replicates + 1 VOI-guided per fraction), evaluate the
    condition hold-out, and locate the fraction that reaches 90% of full-screen
    accuracy (Figure 3).

Everything here is pure/injectable: ``subsampling_curve`` takes a ``train_eval_fn``
(real one wraps Developer 1's models + ``core.eval``; tests pass a closure), so the
VOI math is validated on mock prediction arrays with no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Disagreement
# ---------------------------------------------------------------------------
def _stack_predictions(preds: Dict[str, pd.DataFrame]) -> tuple[np.ndarray, pd.Index]:
    """Align an ensemble of {model: delta_df(index=pert_id, cols=genes)} to a common
    (pert_id x gene) frame and stack to [M, P, G]. Uses the intersection of pert_ids
    and the union of genes (missing genes filled 0, i.e. 'no predicted change')."""
    if len(preds) < 2:
        raise ValueError("ensemble disagreement needs >= 2 models")
    frames = list(preds.values())
    common_index = frames[0].index
    for f in frames[1:]:
        common_index = common_index.intersection(f.index)
    if len(common_index) == 0:
        raise ValueError("ensemble models share no perturbations (empty index intersection)")
    all_genes = frames[0].columns
    for f in frames[1:]:
        all_genes = all_genes.union(f.columns)
    stack = np.stack(
        [f.reindex(index=common_index, columns=all_genes).fillna(0.0).to_numpy(dtype=float) for f in frames],
        axis=0,
    )
    return stack, common_index


def _mean_pairwise_l2(stack: np.ndarray, squared: bool = False) -> np.ndarray:
    """Mean over model pairs of the L2 distance between per-gene prediction vectors.

    stack: [M models, P perts, G genes] -> [P] disagreement per perturbation.
    """
    m = stack.shape[0]
    if m < 2:
        return np.zeros(stack.shape[1])
    acc = np.zeros(stack.shape[1])
    n_pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            diff = stack[i] - stack[j]                       # [P, G]
            d = np.sum(diff * diff, axis=1)                  # squared L2 per pert
            acc += d if squared else np.sqrt(d)
            n_pairs += 1
    return acc / n_pairs


def ensemble_disagreement(
    preds: Dict[str, pd.DataFrame],
    squared: bool = False,
) -> pd.Series:
    """Per-perturbation VOI score (higher = more disagreement = more worth measuring).

    ``preds``: {model_name: delta DataFrame}. Returns a Series indexed by pert_id.
    """
    stack, index = _stack_predictions(preds)
    scores = _mean_pairwise_l2(stack, squared=squared)
    return pd.Series(scores, index=index, name="voi_disagreement")


def rank_perturbations_by_voi(disagreement: pd.Series) -> pd.Series:
    """Perturbations sorted by descending disagreement (highest VOI first)."""
    return disagreement.sort_values(ascending=False)


def select_by_voi(disagreement: pd.Series, fraction: float) -> list:
    """Top ``fraction`` of perturbations by VOI (the informative subset to measure)."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    k = max(1, int(round(fraction * len(disagreement))))
    return list(rank_perturbations_by_voi(disagreement).index[:k])


def gene_disagreement(preds: Dict[str, pd.DataFrame]) -> pd.Series:
    """Per-GENE mean pairwise L2 disagreement (for the biology annotation figure,
    §12: top-20 VOI-disagreement genes). Returns a Series indexed by gene."""
    stack, _index = _stack_predictions(preds)       # [M, P, G]
    m = stack.shape[0]
    # gene columns must match the union alignment used inside _stack_predictions
    frames = list(preds.values())
    genes = frames[0].columns
    for f in frames[1:]:
        genes = genes.union(f.columns)
    acc = np.zeros(stack.shape[2])
    n_pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            diff = stack[i] - stack[j]                       # [P, G]
            acc += np.sqrt(np.mean(diff * diff, axis=0))     # rms over perts, per gene
            n_pairs += 1
    return pd.Series(acc / n_pairs, index=genes, name="gene_disagreement").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 2. Subsampling / sample-efficiency curve
# ---------------------------------------------------------------------------
@dataclass
class SubsamplingCurve:
    fractions: list
    random_mean: list
    random_std: list
    voi_score: list
    full_score: float
    random_90_fraction: Optional[float] = None
    voi_90_fraction: Optional[float] = None
    records: list = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fraction": self.fractions,
                "random_mean": self.random_mean,
                "random_std": self.random_std,
                "voi": self.voi_score,
            }
        )


def _first_fraction_reaching(fractions: Sequence[float], scores: Sequence[float], target: float):
    """Smallest fraction whose score >= target (linear-interpolated between points)."""
    fr = list(fractions)
    sc = list(scores)
    for i, s in enumerate(sc):
        if s >= target:
            if i == 0:
                return fr[0]
            f0, f1, s0, s1 = fr[i - 1], fr[i], sc[i - 1], sc[i]
            if s1 == s0:
                return f1
            return f0 + (f1 - f0) * (target - s0) / (s1 - s0)
    return None


def subsampling_curve(
    train_perturbations: Sequence,
    fractions: Sequence[float],
    train_eval_fn: Callable[[Sequence, int], float],
    voi_scores: Optional[pd.Series] = None,
    n_random_replicates: int = 3,
    seed: int = 42,
    target_ratio: float = 0.90,
) -> SubsamplingCurve:
    """Sample-efficiency curve on the condition hold-out (§7h, claim S1).

    ``train_eval_fn(selected_perts, seed) -> float`` trains the ensemble on the given
    subset of training perturbations and returns the condition-hold-out Pearson-delta.
    For each fraction it is called ``n_random_replicates`` times on random subsets
    plus once on the VOI-guided subset (top-fraction by ``voi_scores``). The
    full-screen score is the fraction==1.0 point. Annotates where random and VOI reach
    ``target_ratio`` of full.
    """
    rng = np.random.default_rng(seed)
    perts = list(train_perturbations)
    n = len(perts)
    fractions = sorted(set(float(f) for f in fractions))

    random_mean, random_std, voi_score, records = [], [], [], []
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        # random replicates
        r_scores = []
        for rep in range(n_random_replicates):
            sub = list(rng.choice(perts, size=k, replace=False))
            s = float(train_eval_fn(sub, seed + rep))
            r_scores.append(s)
            records.append({"fraction": frac, "selection": "random", "replicate": rep, "score": s})
        random_mean.append(float(np.mean(r_scores)))
        random_std.append(float(np.std(r_scores)))
        # VOI-guided
        if voi_scores is not None:
            v_sub = select_by_voi(voi_scores.reindex(perts).dropna(), frac)
            if len(v_sub) < k:  # top-up if VOI covers fewer perts than the fraction asks
                remaining = [p for p in perts if p not in set(v_sub)]
                v_sub = v_sub + list(rng.choice(remaining, size=k - len(v_sub), replace=False)) if remaining else v_sub
            vs = float(train_eval_fn(v_sub, seed))
            voi_score.append(vs)
            records.append({"fraction": frac, "selection": "voi", "replicate": 0, "score": vs})
        else:
            voi_score.append(float("nan"))

    full_score = random_mean[fractions.index(1.0)] if 1.0 in fractions else max(random_mean)
    curve = SubsamplingCurve(
        fractions=list(fractions),
        random_mean=random_mean,
        random_std=random_std,
        voi_score=voi_score,
        full_score=full_score,
        records=records,
    )
    # "90% of full-screen accuracy" is only well-defined when the full-screen anchor is
    # strictly positive. On the hardest (unseen-condition) hold-out the full-ensemble
    # Pearson-delta can be <= 0, in which case target_ratio*full_score would INVERT the
    # threshold (the full point could fail its own target). Leave the 90% marks unset.
    if full_score > 0:
        target = target_ratio * full_score
        curve.random_90_fraction = _first_fraction_reaching(fractions, random_mean, target)
        if voi_scores is not None:
            curve.voi_90_fraction = _first_fraction_reaching(fractions, voi_score, target)
    return curve


# ---------------------------------------------------------------------------
# 3. Production entry: disagreement straight from the run files.
# ---------------------------------------------------------------------------
def disagreement_from_run_files(model_names: Sequence[str], split: str) -> pd.Series:
    """Load ``runs/<model>_<split>.parquet`` for each model and compute per-perturbation
    VOI disagreement. Used after the ensemble's predictions land (Lane C)."""
    from core import contract

    preds = {}
    for m in model_names:
        path = contract.run_path(m, split)
        if path.exists():
            preds[m] = pd.read_parquet(path)
    if len(preds) < 2:
        raise FileNotFoundError(
            f"need >=2 run files for split '{split}' to compute disagreement; found {list(preds)}"
        )
    return ensemble_disagreement(preds)


__all__ = [
    "ensemble_disagreement", "rank_perturbations_by_voi", "select_by_voi",
    "gene_disagreement", "SubsamplingCurve", "subsampling_curve",
    "disagreement_from_run_files",
]
