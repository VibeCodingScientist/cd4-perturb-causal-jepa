"""CausalDGP simulator (keyed to Qi & Chapfuwa, Appendix B).

Linear structural model with Ornstein-Uhlenbeck latent dynamics and a
negative-binomial emission. Two contexts differ in one of four modes
{none, a, b, both}, so cross-context transportability is known by construction.

Ground-truth causal object per context is the influence matrix ``A_C``. For an
intervention with shift vector ``Gamma_q`` the stationary latent effect is

    tau_C^q = mu_pert - mu_ctrl = -A_C^{-1} Gamma_q          (Proposition 1)

which depends only on ``A_C`` and ``Gamma_q`` (the basal input ``b`` cancels).
The estimator (mechanism.py) never sees the true ``A_C``; only labels.py does.

This is CPU-only, pure linear algebra + sampling; each instance runs in seconds.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


# ---------------------------------------------------------------------------
# Influence matrices A_C
# ---------------------------------------------------------------------------
def make_A(G: int, n_reg: int, rng: np.random.Generator, margin: float = 1.0) -> np.ndarray:
    """Sparse weighted influence matrix, stabilized to be Hurwitz.

    Each gene i is regulated by ``n_reg`` other genes with weights drawn from
    [-3, -1] U [1, 3]. The spectrum is then shifted so that max Re(eig) <= -margin,
    guaranteeing a stable OU process with a finite stationary distribution.
    """
    W = np.zeros((G, G))
    for i in range(G):
        regs = rng.choice([j for j in range(G) if j != i], size=n_reg, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_reg)
        W[i, regs] = signs * rng.uniform(1.0, 3.0, size=n_reg)
    lam = np.max(np.real(np.linalg.eigvals(W)))
    A = W - (lam + margin) * np.eye(G)  # Re(eig(A)) <= -margin  => stationary OU
    return A


def rewire(A: np.ndarray, rng: np.random.Generator, frac: float = 0.5) -> np.ndarray:
    """Mode 'a': re-draw a fraction of existing off-diagonal edges, then re-stabilize.

    This changes the causal structure ``A`` (hence the effect map ``-A^{-1}``) while
    keeping the sparsity pattern's scale comparable, so transportability is broken
    in a controlled way.
    """
    A2 = A.copy()
    G = A.shape[0]
    off = [(i, j) for i in range(G) for j in range(G) if i != j and A[i, j] != 0.0]
    n_change = int(frac * len(off))
    for k in rng.choice(len(off), n_change, replace=False):
        i, j = off[k]
        A2[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 3.0)
    lam = np.max(np.real(np.linalg.eigvals(A2)))
    return A2 - (lam + 1.0) * np.eye(G)  # re-stabilize to Re(eig) <= -1


def make_context_pair(G: int, n_reg: int, mode: str, rng: np.random.Generator):
    """Two contexts differing per ``mode`` in {none, a, b, both}.

    - none : identical mechanism            -> every perturbation transportable
    - a    : influence matrix A rewired      -> effects change (blocked)
    - b    : basal input b shifted only      -> A unchanged, effects identical (transportable)
    - both : A rewired and b shifted         -> effects change (blocked)

    Returns ((A_C, b_C), (A_Cp, b_Cp)).
    """
    A = make_A(G, n_reg, rng)
    b = rng.normal(0.0, 1.0, G)
    A_p = rewire(A, rng) if mode in ("a", "both") else A.copy()
    b_p = (b + rng.normal(0.0, 1.0, G)) if mode in ("b", "both") else b.copy()
    return (A, b), (A_p, b_p)


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------
def gamma_for_gene(k: int, G: int, rng: np.random.Generator, mag: float | None = None) -> np.ndarray:
    """Single-coordinate intervention on gene k (CRISPRi knockdown).

    Gamma_q ~= -mag * e_k. Sign is fixed (knockdown = negative shift); the
    magnitude is drawn once per gene and must be reused consistently.
    """
    g = np.zeros(G)
    g[k] = -(mag if mag is not None else rng.uniform(1.0, 3.0))
    return g


# ---------------------------------------------------------------------------
# Stationary moments + emission
# ---------------------------------------------------------------------------
def stationary_moments(A: np.ndarray, b: np.ndarray, gamma: np.ndarray, sigma: float = 0.5):
    """Analytic stationary mean and covariance of the OU process.

    dx = (A x + b + gamma) dt + sigma dW  =>
        mean:  mu = -A^{-1} (b + gamma)
        cov :  A Sigma + Sigma A^T + sigma^2 I = 0

    NOTE (fix vs brief): scipy.solve_continuous_lyapunov(A, Q) solves
    A X + X A^H = Q. To obtain  A Sigma + Sigma A^T = -sigma^2 I  we must pass
    Q = -sigma^2 I, otherwise Sigma comes back negative-definite for Hurwitz A.
    """
    mu = -np.linalg.solve(A, b + gamma)
    D = (sigma ** 2) * np.eye(A.shape[0])
    Sigma = solve_continuous_lyapunov(A, -D)
    Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize away numerical asymmetry
    return mu, Sigma


def emit_counts(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
    theta: float = 5.0,
    libsize: float = 1e4,
    offset: float = 0.0,
) -> np.ndarray:
    """Draw cells around the stationary latent distribution, then NB emission.

    softplus link keeps rates non-negative (offset=0.0 is faithful to the brief's
    ``rate = softplus(x)``; a positive offset would push toward the linear regime
    but empirically lowers the effect SNR, so it is left at 0). Counts are
    compositional (sum-normalized to library size) then Gamma-Poisson (NB).
    """
    x = rng.multivariate_normal(mu, Sigma, size=n_cells)          # (n_cells, G) latent
    rate = np.log1p(np.exp(x + offset))                           # softplus, near-linear
    p = rate / rate.sum(1, keepdims=True)                         # compositional
    lam = p * libsize
    g = rng.gamma(theta, lam / theta)                            # NB = Gamma-Poisson
    return rng.poisson(g)                                         # (n_cells, G) counts


def observed_effect(
    A: np.ndarray,
    b: np.ndarray,
    gamma: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
    sigma: float = 0.5,
    **emit_kw,
) -> np.ndarray:
    """Pseudobulk effect (log space) an estimator would see: mean(pert) - mean(ctrl)."""
    mu_c, S_c = stationary_moments(A, b, np.zeros_like(gamma), sigma=sigma)
    mu_p, S_p = stationary_moments(A, b, gamma, sigma=sigma)
    ctrl = np.log1p(emit_counts(mu_c, S_c, n_cells, rng, **emit_kw).mean(0))
    pert = np.log1p(emit_counts(mu_p, S_p, n_cells, rng, **emit_kw).mean(0))
    return pert - ctrl                                            # (G,) noisy tau_hat


def control_counts(
    A: np.ndarray,
    b: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
    sigma: float = 0.5,
    **emit_kw,
) -> np.ndarray:
    """Raw control-cell count matrix (n_cells, G) for the observational nulls."""
    mu_c, S_c = stationary_moments(A, b, np.zeros(A.shape[0]), sigma=sigma)
    return emit_counts(mu_c, S_c, n_cells, rng, **emit_kw)


# ===========================================================================
# SPIKE #2 -- Nonlinear CausalDGP
# ---------------------------------------------------------------------------
# Saturating drift  dx = ( A h_lambda(x) + b + Gamma ) dt + sigma dW  with
#   h_lambda(x) = (1-lam) x + lam * s * tanh(x/s)   ->   h_0(x) = x  (spike-1 linear system).
# h' in (1-lam, 1] > 0, so A's stabilizing structure (strong negative diagonal) is preserved.
#
# NOTE on the observation model (documented deviation from spike-1): spike-2 works entirely in
# the *latent* x-space -- the nonlinear fixed-point / b_hat machinery is only self-consistent at
# order-1 latent magnitudes, whereas spike-1's log-count operating point (~7) would collapse the
# tanh and wreck the fixed-point solve. So "observed" here = finite-cell pseudobulk mean of latent
# cells drawn around the fixed point with the LOCAL stationary covariance (Lyapunov on the
# fixed-point Jacobian). This keeps the correlation null's linear-regime "free lunch" intact
# (local cov -> Lyapunov(A) at lam=0), so the lam=0 anchor still reproduces spike-1 qualitatively.
# ===========================================================================
from scipy.optimize import fsolve


def h_lambda(x, lam, s=1.5):
    return (1.0 - lam) * x + lam * s * np.tanh(x / s)


def h_lambda_prime(x, lam, s=1.5):
    return (1.0 - lam) + lam * (1.0 - np.tanh(x / s) ** 2)


def fixed_point(A, b, gamma, lam, x0=None, s=1.5):
    """Solve A h_lambda(x) + b + gamma = 0. At lam=0 this is the linear solve -A^{-1}(b+gamma).
    Uses the analytic Jacobian A diag(h') and a linear warm start. Returns (x, converged)."""
    rhs = b + gamma
    x_lin = -np.linalg.solve(A, rhs)
    if lam == 0:
        return x_lin, True
    x0 = x_lin if x0 is None else x0
    f = lambda x: A @ h_lambda(x, lam, s) + rhs
    jac = lambda x: A * h_lambda_prime(x, lam, s)[None, :]  # A @ diag(h'(x))
    sol, _info, ier, _msg = fsolve(f, x0, fprime=jac, full_output=True)
    converged = (ier == 1) and np.all(np.isfinite(sol)) and (np.linalg.norm(f(sol)) < 1e-6)
    return sol, converged


def is_stable(A, x_star, lam, s=1.5):
    """Fixed-point Jacobian J = A diag(h'(x*)) must be Hurwitz (all Re(eig) < 0)."""
    J = A * h_lambda_prime(x_star, lam, s)[None, :]
    return np.max(np.real(np.linalg.eigvals(J))) < 0


def stationary_cov_local(A, x_star, lam, sigma=0.5, s=1.5):
    """Local stationary covariance: solve J Sigma + Sigma J^T = -sigma^2 I at J = A diag(h'(x*))."""
    J = A * h_lambda_prime(x_star, lam, s)[None, :]
    D = (sigma ** 2) * np.eye(A.shape[0])
    Sigma = solve_continuous_lyapunov(J, -D)
    return 0.5 * (Sigma + Sigma.T)


def true_effect_nl(A, b, gamma, lam, s=1.5):
    """Noiseless nonlinear effect tau = x*(gamma) - x*(0) (for labels / oracle / epistasis)."""
    x_c, ok_c = fixed_point(A, b, np.zeros_like(gamma), lam, s=s)
    x_p, ok_p = fixed_point(A, b, gamma, lam, x0=x_c, s=s)
    return x_p - x_c, (ok_c and ok_p)


def latent_cells(x_star, Sigma, n_cells, rng):
    """Draw n_cells latent cells ~ N(x*, Sigma) (the spike-2 'emission')."""
    return rng.multivariate_normal(x_star, Sigma, size=n_cells)
