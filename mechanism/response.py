"""C-NL gate -- third-moment fluctuation-response machinery.

Tests whether the baseline third moment T_ijk = <dX_i dX_j dX_k>_0 of unperturbed fluctuations
predicts the component of the perturbation response that the first-order (CIPHER) model Sigma u leaves
unexplained: r = Delta_mu - Sigma u, hypothesized ~ c * T[u,u].

PROVENANCE (see FINDINGS_CNL.md / brief section 0): the third-moment link is an INFERENCE from
standard fluctuation-response theory, NOT a CIPHER result. T[u,u] is a *candidate* predictor; its
coefficient is FIT, never assumed to be the analytic 1/2. Nothing here attributes the breakdown to a
third moment "because CIPHER says so".
"""

from __future__ import annotations
import numpy as np


def empirical_third_moment(control_latent: np.ndarray) -> np.ndarray:
    """Baseline third central moment T_ijk = <dX_i dX_j dX_k>_0 from control fluctuations.

    control_latent: (n_cells, G). Computed with a memory-safe loop over k (the stub's
    einsum('ni,nj,nk->ijk') would allocate an ~(n, G, G) = multi-GB intermediate at n~1e5)."""
    dX = control_latent - control_latent.mean(0)
    n, G = dX.shape
    T = np.empty((G, G, G))
    for k in range(G):
        Wk = dX * dX[:, k:k + 1]          # (n, G): each row scaled by dX_nk
        T[:, :, k] = Wk.T @ dX / n        # (G, G): mean_n dX_i dX_j dX_k
    return T


def second_order_term(T: np.ndarray, u: np.ndarray) -> np.ndarray:
    """T[u,u]_i = sum_jk T_ijk u_j u_k -- the third-moment contraction (second-order response feature)."""
    return np.tensordot(np.tensordot(T, u, axes=([2], [0])), u, axes=([1], [0]))


def first_order_residual(dmu: np.ndarray, Sigma: np.ndarray, u: np.ndarray) -> np.ndarray:
    """r = Delta_mu - Sigma u -- the part CIPHER's linear model leaves unexplained."""
    return dmu - Sigma @ u


def covariance_surrogate(Sigma: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Second-moment (correlation-only) surrogate feature for the residual, quadratic in u:
    B[u]_i = (Sigma u)_i^2. This is the best a covariance-only method has for an O(u^2) response term
    (the Gaussian connected third moment is exactly 0); it is the additive/correlation baseline the
    third moment must beat."""
    su = Sigma @ u
    return su * su


def r2_fit(y: np.ndarray, x: np.ndarray) -> float:
    """R^2 of predicting y from a single feature x with a FIT coefficient (centered => intercept).
    Coefficient is learned (not assumed) per the provenance rule."""
    y = y.reshape(-1).astype(float)
    x = x.reshape(-1).astype(float)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(xc @ xc)
    if denom <= 0:
        return 0.0
    c = float(xc @ yc) / denom
    pred = c * xc
    ss_res = float(np.sum((yc - pred) ** 2))
    ss_tot = float(np.sum(yc ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def delta_r2(residual_stack, feat_stack, baseline_stack) -> float:
    """Variance of the first-order residual explained by the third-moment feature T[u,u] minus that
    explained by the covariance surrogate. Pooled over perturbations and genes. Pass: > 0 with an
    >=8-seed cluster-bootstrap CI excluding 0."""
    return r2_fit(residual_stack, feat_stack) - r2_fit(residual_stack, baseline_stack)
