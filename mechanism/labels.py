"""Analytic transportability labels (Proposition 1).

Ground truth computed from the TRUE per-context influence matrices A_C, A_Cp.
This module is the ONLY place the true matrices may be read; the estimator in
mechanism.py must never see them.

A perturbation q (shift Gamma_q) is transportable C <-> C' iff its stationary
effects agree:  tau_C^q = tau_Cp^q, with tau = -A^{-1} Gamma. We report a
continuous cosine agreement s in [-1, 1] and binarize at ``thresh``.
"""

from __future__ import annotations

import numpy as np


def true_effect(A: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Latent stationary effect of intervention gamma: tau = -A^{-1} gamma."""
    return -np.linalg.solve(A, gamma)


def true_transport_label(A_C, A_Cp, gamma_star, thresh: float = 0.9):
    """Return (binary_label, continuous_agreement) for one perturbation.

    continuous_agreement = cos(tau_C, tau_Cp) in [-1, 1]; binary = 1 if >= thresh.
    """
    tC = true_effect(A_C, gamma_star)
    tCp = true_effect(A_Cp, gamma_star)
    denom = np.linalg.norm(tC) * np.linalg.norm(tCp)
    s = float(tC @ tCp / denom) if denom > 0 else 0.0
    return int(s >= thresh), s


# ---------------------------------------------------------------------------
# SPIKE #2 -- nonlinear transportability label + linear oracle
# ---------------------------------------------------------------------------
def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def true_transport_label_nl(A_C, b_C, A_Cp, b_Cp, gamma_star, lam, thresh=0.9, s=0.4):
    """Ground-truth transportability from the true NONLINEAR effects across contexts.

    Returns (binary_label, continuous_agreement, ok). `ok` is False if either fixed point failed
    to converge -- the caller should drop that perturbation. Under nonlinearity even mode `b`
    (basal shift, same A) is no longer perfectly transportable: the shift moves the operating point
    into a different saturation regime, so the effective response changes -- correct and expected.
    """
    from causaldgp import true_effect_nl
    tC, okC = true_effect_nl(A_C, b_C, gamma_star, lam, s=s)
    tCp, okCp = true_effect_nl(A_Cp, b_Cp, gamma_star, lam, s=s)
    if not (okC and okCp):
        return 0, 0.0, False
    agree = _cos(tC, tCp)
    return int(agree >= thresh), agree, True


def linear_oracle_transport(A_C, A_Cp, gamma_star):
    """Diagnostic: transportability predicted by the LINEAR condition -A_true^{-1} Gamma with the
    true A (operating-point-blind). At lam=0 it equals the label; as lam grows it drifts from the
    nonlinear truth -- quantifying how far the field's linear transportability condition strays."""
    return _cos(true_effect(A_C, gamma_star), true_effect(A_Cp, gamma_star))
