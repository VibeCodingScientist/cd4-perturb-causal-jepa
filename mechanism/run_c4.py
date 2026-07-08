"""C4 experiment harness: modes x seeds -> per-perturbation records.

For each (divergence mode, seed) it builds a context pair, splits genes into a
training set (perturbed) and a held-out set (perturbed in NEITHER context), then:
  - estimates a per-context influence matrix A_hat from interventional constraints,
  - predicts each held-out perturbation's cross-context transportability with the
    mechanism estimator and the two correlation-only nulls,
  - records the analytic ground-truth transportability (from the true A matrices).

Run:  python run_c4.py         -> writes results/c4_records.csv
Then: python eval.py           -> writes results/c4_auroc.csv + results/moneyshot.png
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from causaldgp import make_context_pair, gamma_for_gene, observed_effect, control_counts
from mechanism import estimate_A, transport_score, corr_null_scores, gears_null_scores
from labels import true_transport_label


def _cos(a, b) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0

MODES = ["none", "a", "b", "both"]

# -- Pre-registered configuration (fixed BEFORE inspecting any transportability
# -- label; alpha/ridge chosen by a coarse sweep on a disjoint dev-seed set; see
# -- FINDINGS.md "Hyperparameters"). Do not tune after seeing labels. ----------
CONFIG = dict(
    G=50,            # genes
    n_reg=6,         # regulators per gene
    P_train=30,      # perturbed (training) genes  -> P < G forces extrapolation
    P_holdout=15,    # held-out genes, perturbed in neither context
    n_cells=1000,    # cells per (context, perturbation); 1000 gives a stable M0 (r>0.9)
    sigma=0.5,       # OU latent noise
    libsize=1e4,     # NB library size
    theta=5.0,       # NB dispersion
    mag_lo=1.0,      # knockdown magnitude range (fixed per gene)
    mag_hi=3.0,
    alpha=0.002,     # Lasso sparsity: dev-sweep optimum for A-recovery (cos(A_hat,A_true)~0.80).
                     # The brief's illustrative 0.05 over-shrinks to cos~0.29 (mechanism at chance);
                     # 0.002 gives the mechanism its best honest shot. Chosen on a disjoint dev-seed
                     # set by maximizing A-recovery (label-independent), NOT by maximizing any gap.
    ridge=1e-3,      # ridge stabilization in predict_effect
    thresh=0.9,      # transportable iff cos(tau_C, tau_Cp) >= thresh
    knn=10,          # neighbors for the GEARS-style null
    n_seeds=8,       # seeds per mode (>= 5 required)
    seed_base=700,   # base RNG seed for the reported run (dev sweep uses a disjoint base)
)


def _seed_for(mode: str, seed: int, seed_base: int) -> int:
    """Distinct, deterministic RNG seed per (mode, seed)."""
    return seed_base + MODES.index(mode) * 10_000 + seed


def run_instance(mode: str, seed: int, cfg: dict = CONFIG) -> list[dict]:
    """One (mode, seed) instance -> list of per-held-out-perturbation records."""
    G, n_reg = cfg["G"], cfg["n_reg"]
    rng = np.random.default_rng(_seed_for(mode, seed, cfg["seed_base"]))

    (A_C, b_C), (A_Cp, b_Cp) = make_context_pair(G, n_reg, mode, rng)

    perm = rng.permutation(G)
    train_genes = list(perm[: cfg["P_train"]])
    holdout_genes = list(perm[cfg["P_train"]: cfg["P_train"] + cfg["P_holdout"]])

    # Fixed knockdown magnitude per gene, reused everywhere (train, holdout, label).
    mag_by_gene = {int(k): float(rng.uniform(cfg["mag_lo"], cfg["mag_hi"])) for k in range(G)}
    gamma = {k: gamma_for_gene(k, G, rng, mag=mag_by_gene[k]) for k in range(G)}

    emit_kw = dict(n_cells=cfg["n_cells"], sigma=cfg["sigma"],
                   libsize=cfg["libsize"], theta=cfg["theta"])

    # Observed (noisy) training effects per context.
    tau_train_C = np.stack([observed_effect(A_C, b_C, gamma[k], rng=rng, **emit_kw) for k in train_genes])
    tau_train_Cp = np.stack([observed_effect(A_Cp, b_Cp, gamma[k], rng=rng, **emit_kw) for k in train_genes])
    gamma_train = np.stack([gamma[k] for k in train_genes])

    # --- Mechanism: estimate A from interventional constraints ---
    A_hat_C = estimate_A(tau_train_C, gamma_train, alpha=cfg["alpha"])
    A_hat_Cp = estimate_A(tau_train_Cp, gamma_train, alpha=cfg["alpha"])

    # --- Nulls: control counts + co-expression graphs (observational) ---
    ctrl_C = control_counts(A_C, b_C, rng=rng, **emit_kw)
    ctrl_Cp = control_counts(A_Cp, b_Cp, rng=rng, **emit_kw)
    coexpr_C = np.corrcoef(np.log1p(ctrl_C), rowvar=False)
    coexpr_Cp = np.corrcoef(np.log1p(ctrl_Cp), rowvar=False)

    corr_scores = corr_null_scores(ctrl_C, ctrl_Cp, holdout_genes, mag_by_gene)
    gears_scores = gears_null_scores(tau_train_C, tau_train_Cp, train_genes, holdout_genes,
                                     coexpr_C, coexpr_Cp, knn=cfg["knn"])

    records = []
    for k in holdout_genes:
        g = gamma[k]
        label, agree = true_transport_label(A_C, A_Cp, g, thresh=cfg["thresh"])
        records.append(dict(
            mode=mode, seed=seed, gene=int(k),
            label=label, agree=agree,
            # brief's exact predictor: transportability = cosine of A_hat^{-1}-solved effects
            mechanism=transport_score(A_hat_C, A_hat_Cp, g, ridge=cfg["ridge"]),
            # principled variant: cosine of the estimated causal-target column k (identifiable
            # for a held-out gene even though its own regulatory row is not) -- avoids the
            # rank-deficient inversion pathology
            mechanism_col=_cos(A_hat_C[:, k], A_hat_Cp[:, k]),
            corr_null=corr_scores[k],
            gears_null=gears_scores[k],
            # oracle ceiling: same column-k score computed from the TRUE matrices (shows the
            # mechanism signal is real; not a predictor, a reference)
            true_col=_cos(A_C[:, k], A_Cp[:, k]),
        ))
    return records


def m0_check(cfg: dict = CONFIG, n_genes: int = 20, n_instances: int = 3) -> float:
    """M0 milestone: the empirical pseudobulk effect must correlate (Pearson r > 0.9) with the
    analytic -A^{-1}Gamma, else the simulator/emission is wrong and nothing downstream is valid."""
    from labels import true_effect
    G, n_reg = cfg["G"], cfg["n_reg"]
    emit_kw = dict(n_cells=cfg["n_cells"], sigma=cfg["sigma"], libsize=cfg["libsize"], theta=cfg["theta"])
    emp, ana = [], []
    for s in range(n_instances):
        rng = np.random.default_rng(cfg["seed_base"] + 50_000 + s)
        (A, b), _ = make_context_pair(G, n_reg, "none", rng)
        for k in rng.choice(G, n_genes, replace=False):
            g = gamma_for_gene(k, G, rng, mag=rng.uniform(cfg["mag_lo"], cfg["mag_hi"]))
            emp.append(observed_effect(A, b, g, rng=rng, **emit_kw))
            ana.append(true_effect(A, g))
    r = float(np.corrcoef(np.concatenate(emp), np.concatenate(ana))[0, 1])
    print(f"M0 simulator check: empirical vs analytic effect  Pearson r = {r:.3f}  "
          f"({'PASS' if r > 0.9 else 'FAIL'}, need > 0.9)")
    return r


def generate(cfg: dict = CONFIG, modes=MODES, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for mode in modes:
        for seed in range(cfg["n_seeds"]):
            rows.extend(run_instance(mode, seed, cfg))
        if verbose:
            print(f"  mode={mode:5s} done ({cfg['n_seeds']} seeds)")
    return pd.DataFrame(rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    m0_check(CONFIG)
    print(f"Running C4 sweep: {len(MODES)} modes x {CONFIG['n_seeds']} seeds "
          f"x {CONFIG['P_holdout']} held-out perturbations ...")
    df = generate(CONFIG)
    out = os.path.join(out_dir, "c4_records.csv")
    df.to_csv(out, index=False)
    n_pos = int(df["label"].sum())
    print(f"\nWrote {out}  ({len(df)} records; {n_pos} transportable / {len(df) - n_pos} blocked)")
    print("Next: python eval.py")


if __name__ == "__main__":
    main()
