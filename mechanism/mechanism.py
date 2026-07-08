"""Mechanism estimator (interventional) and two correlation-only nulls.

All three methods output, per held-out perturbation, a predicted transportability
score in [-1, 1] (cosine of the predicted effect across the two contexts). Only
the mechanism estimator uses the interventional constraint A tau_q = -Gamma_q;
the nulls use correlation structure only. C4 asks whether the interventional
structure buys better recovery of the ground-truth transportability label.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso


# ===========================================================================
# MECHANISM ESTIMATOR: sparse A from interventional constraints  A tau_q = -Gamma_q
# ===========================================================================
def estimate_A(tau_train: np.ndarray, gamma_train: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Estimate the influence matrix row-by-row via sparse linear regression.

    tau_train, gamma_train: (P, G) observed effect and intervention vectors.
    Row i of A is the sparse a_i minimizing
        sum_q (a_i . tau_q + Gamma_q[i])^2 + alpha |a_i|_1 .
    Rows whose gene was never intervened on (Gamma_q[i]==0 for all q) have an
    all-zero target and are returned as zero rows -- this is exactly the
    identifiability limit: A_C is only pinned down along observed intervention
    directions (see FINDINGS / brief section 4).
    """
    P, G = tau_train.shape
    A = np.zeros((G, G))
    X = tau_train  # (P, G): rows are tau_q
    for i in range(G):
        y = -gamma_train[:, i]  # (P,): -Gamma_q[i]
        if not np.any(y):
            continue  # unconstrained row -> leave at zero
        A[i, :] = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000).fit(X, y).coef_
    return A


def predict_effect(A_hat: np.ndarray, gamma_star: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Predicted effect of a (held-out) perturbation: solve A_hat tau = -Gamma_star.

    Ridge-stabilized because A_hat is rank-deficient (unconstrained held-out rows
    are zero); the ridge makes the solve well-posed.
    """
    G = A_hat.shape[0]
    return np.linalg.solve(A_hat + ridge * np.eye(G), -gamma_star)


def transport_score(A_hat_C: np.ndarray, A_hat_Cp: np.ndarray, gamma_star: np.ndarray,
                    ridge: float = 1e-3) -> float:
    """Predicted transportability in [-1, 1] = cosine of predicted effects across contexts."""
    tC = predict_effect(A_hat_C, gamma_star, ridge)
    tCp = predict_effect(A_hat_Cp, gamma_star, ridge)
    d = np.linalg.norm(tC) * np.linalg.norm(tCp)
    return float(tC @ tCp / d) if d > 0 else 0.0


# ===========================================================================
# NULL 1: correlation graph -- effect column predicted from the control covariance
# ===========================================================================
def corr_null_scores(ctrl_counts_C: np.ndarray, ctrl_counts_Cp: np.ndarray,
                     holdout_genes, mag_by_gene: dict) -> dict:
    """Predict a held-out effect as the (scaled) covariance column with the perturbed
    gene: tau ~ -Sigma[:, k] * mag. Purely observational (no interventions). Score
    transportability = cosine of that prediction across the two contexts.
    """
    SC = np.cov(np.log1p(ctrl_counts_C), rowvar=False)
    SCp = np.cov(np.log1p(ctrl_counts_Cp), rowvar=False)
    out = {}
    for k in holdout_genes:
        m = mag_by_gene[k]
        tC, tCp = -SC[:, k] * m, -SCp[:, k] * m
        d = np.linalg.norm(tC) * np.linalg.norm(tCp)
        out[k] = float(tC @ tCp / d) if d > 0 else 0.0
    return out


# ===========================================================================
# NULL 2: co-expression propagation ("GEARS-style")
# ===========================================================================
def _coexpr_predict(tau_train: np.ndarray, coexpr: np.ndarray, train_genes, k: int,
                    knn: int) -> np.ndarray:
    """Predict held-out gene k's effect vector as a co-expression-weighted average of
    observed training-gene effect vectors (kNN label propagation on the co-expression
    graph). Uses the graph but no interventional structure learning.
    """
    sims = np.array([coexpr[k, j] for j in train_genes])  # similarity of gene k to each train gene
    kk = min(knn, len(train_genes))
    order = np.argsort(-np.abs(sims))[:kk]                 # nearest neighbors by |co-expression|
    w = sims[order]
    denom = np.sum(np.abs(w))
    if denom == 0:
        return np.zeros(tau_train.shape[1])
    w = w / denom
    return w @ tau_train[order]                            # (G,) predicted effect


def gears_null_scores(tau_train_C: np.ndarray, tau_train_Cp: np.ndarray,
                      train_genes, holdout_genes, coexpr_C: np.ndarray, coexpr_Cp: np.ndarray,
                      knn: int = 10) -> dict:
    """Co-expression label propagation. For each held-out gene predict its effect in
    each context from the co-expression-weighted training effects, then score
    transportability as the cosine of the two predictions.

    coexpr_*: (G, G) control-cell correlation matrices. tau_train_* rows align with
    ``train_genes`` order.
    """
    train_genes = list(train_genes)
    out = {}
    for k in holdout_genes:
        tC = _coexpr_predict(tau_train_C, coexpr_C, train_genes, k, knn)
        tCp = _coexpr_predict(tau_train_Cp, coexpr_Cp, train_genes, k, knn)
        d = np.linalg.norm(tC) * np.linalg.norm(tCp)
        out[k] = float(tC @ tCp / d) if d > 0 else 0.0
    return out


# ===========================================================================
# SPIKE #2 -- nonlinear double-perturbation predictors
# ===========================================================================
def estimate_b(A_hat, ctrl_mean, lam, s=0.4):
    """b_hat from the observed control operating point: A_hat h_lambda(x*_ctrl) + b = 0."""
    from causaldgp import h_lambda
    return -A_hat @ h_lambda(ctrl_mean, lam, s)


def predict_double_mech(A_hat, b_hat, gamma_i, gamma_j, lam, s=0.4, x_c=None):
    """Mechanism prediction of a DOUBLE's effect: solve the nonlinear fixed point with the estimated
    A, b and the KNOWN nonlinearity (lam, s). Captures epistasis the additive nulls cannot. Falls
    back to the linear prediction if the nonlinear solve fails to converge."""
    from causaldgp import fixed_point
    if x_c is None:
        x_c, okc = fixed_point(A_hat, b_hat, np.zeros_like(gamma_i), lam, s=s)
        if not okc:
            x_c = -np.linalg.solve(A_hat, b_hat)
    x_ij, ok = fixed_point(A_hat, b_hat, gamma_i + gamma_j, lam, x0=x_c, s=s)
    if not ok:  # fall back to the additive/linear prediction
        x_ij = -np.linalg.solve(A_hat, b_hat + gamma_i + gamma_j)
    return x_ij - x_c


def transport_score_double(A_hat_C, b_hat_C, A_hat_Cp, b_hat_Cp, gi, gj, lam, s=0.4):
    tC = predict_double_mech(A_hat_C, b_hat_C, gi, gj, lam, s=s)
    tCp = predict_double_mech(A_hat_Cp, b_hat_Cp, gi, gj, lam, s=s)
    d = np.linalg.norm(tC) * np.linalg.norm(tCp)
    return float(tC @ tCp / d) if d > 0 else 0.0


def additive_null_double(single_effect_C: dict, single_effect_Cp: dict, i, j) -> float:
    """Additive null: double effect = sum of single-effect predictions; cosine across contexts.
    Misses epistasis by construction. single_effect_*: gene -> predicted single-effect vector."""
    tC = single_effect_C[i] + single_effect_C[j]
    tCp = single_effect_Cp[i] + single_effect_Cp[j]
    d = np.linalg.norm(tC) * np.linalg.norm(tCp)
    return float(tC @ tCp / d) if d > 0 else 0.0


def corr_single_effects(ctrl_cells: np.ndarray, genes, mag_by_gene: dict) -> dict:
    """Correlation-additive single predictor: tau_k ~ -Sigma[:,k]*mag from the control covariance."""
    S = np.cov(ctrl_cells, rowvar=False)
    return {int(k): -S[:, k] * mag_by_gene[int(k)] for k in genes}


def gears_single_effects(tau_train: np.ndarray, train_genes, coexpr: np.ndarray, knn: int = 10) -> dict:
    """GEARS-additive single predictor: each gene's single effect from a leave-one-out
    co-expression-weighted average of the OTHER training-gene effects."""
    train_genes = list(train_genes)
    pos = {int(g): r for r, g in enumerate(train_genes)}
    out = {}
    for k in train_genes:
        others = [g for g in train_genes if g != k]
        sims = np.array([coexpr[k, g] for g in others])
        kk = min(knn, len(others))
        order = np.argsort(-np.abs(sims))[:kk]
        w = sims[order]
        denom = np.sum(np.abs(w))
        if denom == 0:
            out[int(k)] = np.zeros(tau_train.shape[1])
        else:
            rows = [pos[others[o]] for o in order]
            out[int(k)] = (w / denom) @ tau_train[rows]
    return out
