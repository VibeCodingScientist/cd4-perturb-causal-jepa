"""
core.models.baselines — Ridge, TabPFN, PseudoBulk-FCN (UNIFIED_BUILD_PLAN.md §7c).

Every baseline predicts a per-perturbation DELTA vector over the HVGs from that
perturbation's gene-token features (ESM-2 ⊕ context prior), writes
`runs/<name>_<split>.parquet`, and is scored by the shared `core.eval`. Training target =
per-perturbation mean delta over the TRAIN pseudobulk rows (condition-agnostic; the honest
weakness a causal/JEPA model must beat on the condition hold-out).

Ridge runs anywhere (numpy/sklearn) and is unit-tested on synthetic data. TabPFN and the
FCN need their own deps (`tabpfn`, `torch`) and are authored to the bioRxiv
2026.06.28.735106 protocol but gated so importing this module never requires them.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import contract as C
from .. import eval as ev
from .. import features as feat


# ---------------------------------------------------------------------------
# Shared data prep
# ---------------------------------------------------------------------------
def train_pert_targets(train_pb: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-perturbation mean delta over the train rows (index = pert_id, cols = HVG)."""
    if train_pb is None:
        train_pb = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    delta = C.pseudobulk_delta(train_pb)
    y = delta.groupby(level="pert_id").mean()
    return y.drop(index=C.CONTROL_PERT_ID, errors="ignore")


def split_test_perts(split: str, pseudobulk_test: Optional[pd.DataFrame] = None) -> List[str]:
    """The canonical (HVG-panel) perturbations to predict for a split — shared with the causal
    model so every benchmark row is scored on the same set (see core.eval.evaluable_perts)."""
    return ev.evaluable_perts(split, pseudobulk_test=pseudobulk_test)


def control_profile(train_pb: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Mean control expression profile over the train rows (HVG-dim) — the FCN's baseline input."""
    if train_pb is None:
        train_pb = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    expr = C.pseudobulk_expr(train_pb)
    ctrl = expr.xs(C.CONTROL_PERT_ID, level="pert_id")
    return ctrl.mean(axis=0).to_numpy(dtype=float)


def _write_and_score(pred: pd.DataFrame, split: str, model_name: str, record: bool):
    pred.to_parquet(C.run_path(model_name, split))
    if record:
        return ev.evaluate_and_record(pred, split, model_name)
    return None


# ---------------------------------------------------------------------------
# Ridge (canonical comparator; CPU)
# ---------------------------------------------------------------------------
def run_ridge(splits: Sequence[str] = C.SPLITS, *, alpha: float = 10.0,
              use_context: bool = True, record: bool = True) -> None:
    """Multi-output Ridge: gene features -> delta(HVG). Fit once, predict every split."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    y = train_pert_targets()
    X = feat.pert_gene_features(y.index, use_context=use_context)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X.to_numpy(), y.to_numpy())

    genes = list(y.columns)
    for split in splits:
        perts = split_test_perts(split)
        Xte = feat.pert_gene_features(perts, use_context=use_context)
        pred = pd.DataFrame(model.predict(Xte.to_numpy()), index=perts, columns=genes)
        pred.index.name = "pert_id"
        _write_and_score(pred, split, C.MODEL_RIDGE, record)


# ---------------------------------------------------------------------------
# PseudoBulk-FCN (3-layer residual MLP; needs torch)
# ---------------------------------------------------------------------------
def run_fcn(splits: Sequence[str] = C.SPLITS, *, hidden: int = 512, epochs: int = 200,
            lr: float = 1e-3, use_context: bool = True, record: bool = True,
            device: Optional[str] = None) -> None:
    """VCC-2nd-place-style FCN: predict a residual delta on top of the control profile
    from [gene features ⊕ control profile]. ~1h; short GPU slot or CPU."""
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover - env gate
        raise RuntimeError("FCN needs `torch`; install it (CPU is fine) or run via gpu_queue.") from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    y = train_pert_targets()
    Xg = feat.pert_gene_features(y.index, use_context=use_context)
    ctrl = control_profile()
    genes = list(y.columns)

    def assemble(perts):
        Xg_ = feat.pert_gene_features(perts, use_context=use_context).to_numpy()
        ctrl_tile = np.tile(ctrl, (len(perts), 1))
        return np.concatenate([Xg_, ctrl_tile], axis=1)

    Xtr = assemble(list(y.index))
    Ytr = y.to_numpy()
    d_in, d_out = Xtr.shape[1], Ytr.shape[1]

    class FCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(), nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
                nn.Linear(hidden, d_out),
            )

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(C.SPLIT_SEED)
    model = FCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    for split in splits:
        perts = split_test_perts(split)
        Xte = torch.tensor(assemble(perts), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(Xte).cpu().numpy()
        pred = pd.DataFrame(out, index=perts, columns=genes)
        pred.index.name = "pert_id"
        _write_and_score(pred, split, C.MODEL_FCN, record)


# ---------------------------------------------------------------------------
# TabPFN (per-target foundation model; needs tabpfn v2/v3)
# ---------------------------------------------------------------------------
def run_tabpfn(splits: Sequence[str] = C.SPLITS, *, n_targets: int = C.TOP_DEG_N,
               max_features: int = C.TABPFN_MAX_FEATURES, record: bool = True,
               device: Optional[str] = None) -> None:
    """One TabPFN regressor per top-DEG-frequency gene (bioRxiv 2026.06.28.735106 protocol).

    Features = ESM-2(perturbed gene) PCA ⊕ global DEG-frequency vector, capped at
    `max_features` (200 for v3 / 500 for v2). The n_targets most-frequently-DE genes are
    modeled by TabPFN; the remaining HVGs are filled with the per-gene training-mean delta so
    the prediction is a full delta vector the frozen eval can score. Respect the row ceiling.
    """
    try:
        from tabpfn import TabPFNRegressor
    except Exception as e:  # pragma: no cover - env gate
        raise RuntimeError(
            "TabPFN needs `tabpfn` (v2 or v3 — NOT the v1 limits); install on the box."
        ) from e
    from sklearn.decomposition import PCA

    y = train_pert_targets()
    if len(y) > C.TABPFN_MAX_ROWS:
        y = y.sample(C.TABPFN_MAX_ROWS, random_state=C.SPLIT_SEED)
    genes = list(y.columns)

    # top-n_targets genes by training DE frequency
    deg = feat.load_deg_freq()
    target_genes = [g for g in deg.columns if g in genes][:n_targets]
    fill_mean = y.mean(axis=0)  # per-gene training-mean delta for the unmodeled genes

    # ESM-2 PCA feature (leave room for the DEG-freq block within the ceiling)
    esm2 = feat.load_esm2()
    deg_dim = deg.shape[1]
    pca_dim = max(2, max_features - deg_dim)
    Xesm_tr = esm2.reindex(y.index).fillna(0.0).to_numpy()
    pca = PCA(n_components=min(pca_dim, Xesm_tr.shape[1]), random_state=C.SPLIT_SEED).fit(Xesm_tr)
    deg_global = deg.mean(axis=0).to_numpy()  # constant context block

    def feats(perts):
        Xe = pca.transform(esm2.reindex(perts).fillna(0.0).to_numpy())
        Xd = np.tile(deg_global, (len(perts), 1))
        return np.concatenate([Xe, Xd], axis=1)

    Xtr = feats(list(y.index))
    device = device or ("cuda" if _torch_cuda() else "cpu")
    regressors = {}
    for g in target_genes:
        reg = TabPFNRegressor(device=device)
        reg.fit(Xtr, y[g].to_numpy())
        regressors[g] = reg

    for split in splits:
        perts = split_test_perts(split)
        Xte = feats(perts)
        pred = pd.DataFrame(
            np.tile(fill_mean.to_numpy(), (len(perts), 1)), index=perts, columns=genes,
        )
        for g in target_genes:
            pred[g] = regressors[g].predict(Xte)
        pred.index.name = "pert_id"
        _write_and_score(pred, split, C.MODEL_TABPFN, record)


def _torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
