"""Spike #2 -- does the mechanism win once the mechanism is nonlinear?

Tests the mechanism-vs-correlation gap on held-out DOUBLE-perturbation transportability as a
function of a single nonlinearity knob lambda. At lam=0 the system is linear, doubles are additive,
and correlation should win (reproducing spike #1 -- the anchor). As lam grows, epistasis appears and
the additive/correlation prediction should break. Does the gap cross zero?

Kill probe runs first (lam in {0, 0.85}); the full grid only runs if the gap moved.

Run:  python run_spike2.py        -> results/spike2_records.csv, results/gap_vs_lambda.{csv,png}
"""

from __future__ import annotations
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causaldgp import (make_context_pair, gamma_for_gene, fixed_point, is_stable,
                       stationary_cov_local, true_effect_nl, latent_cells, h_lambda)
from mechanism import (estimate_A, estimate_b, transport_score_double, additive_null_double,
                       corr_single_effects, gears_single_effects)
from labels import true_transport_label_nl, linear_oracle_transport

MODES = ["none", "a", "b", "both"]
METHODS = ["mechanism", "corr_add", "gears_add", "obs_add", "lin_oracle"]
PRETTY = {"mechanism": "Mechanism (nonlinear Â)", "corr_add": "Null: correlation-additive",
          "gears_add": "Null: GEARS-additive", "obs_add": "Null: observed-additive",
          "lin_oracle": "Linear oracle (true A)"}

CONFIG = dict(
    G=50, n_reg=6, P_train=30, n_pairs=25,   # test doubles = pairs of individually-perturbed genes
    n_cells=1000, sigma=0.5, mag_lo=1.0, mag_hi=3.0,
    alpha=0.002,          # carried from spike #1 (do not re-tune on labels)
    thresh=0.9, knn=10, s=0.4,   # s: saturation scale (manipulation-check calibrated; see FINDINGS)
    n_seeds_probe=5, n_seeds_full=8,
    lam_high=0.85, lam_grid=[0.0, 0.25, 0.5, 0.7, 0.85, 1.0],
    struct_base=1300, noise_base=4700,   # lambda-independent seeds -> paired across lambda
    n_boot=2000, boot_seed=777,
)


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _skey(base, mode, seed):
    return base + MODES.index(mode) * 1000 + seed


def _context_bundle(A, b, train_genes, gamma, mag, lam, cfg, noise_rng):
    """Estimate Â, b̂ and the additive single-effect predictors for one context at one lambda.
    Returns None if the control fixed point is non-convergent or unstable."""
    G, s = cfg["G"], cfg["s"]
    x_c, okc = fixed_point(A, b, np.zeros(G), lam, s=s)
    if not okc or not is_stable(A, x_c, lam, s=s):
        return None
    S_c = stationary_cov_local(A, x_c, lam, sigma=cfg["sigma"], s=s)
    ctrl_cells = latent_cells(x_c, S_c, cfg["n_cells"], noise_rng)
    ctrl_mean = ctrl_cells.mean(0)

    dh, tau_obs, true_single = [], {}, {}
    for k in train_genes:
        x_p, okp = fixed_point(A, b, gamma[k], lam, x0=x_c, s=s)
        if not okp:
            return None
        S_p = stationary_cov_local(A, x_p, lam, sigma=cfg["sigma"], s=s)
        pert_mean = latent_cells(x_p, S_p, cfg["n_cells"], noise_rng).mean(0)
        tau_obs[int(k)] = pert_mean - ctrl_mean
        dh.append(h_lambda(pert_mean, lam, s) - h_lambda(ctrl_mean, lam, s))  # A * Delta_h = -Gamma
        true_single[int(k)] = x_p - x_c
    dh = np.stack(dh)
    gamma_train = np.stack([gamma[k] for k in train_genes])

    A_hat = estimate_A(dh, gamma_train, alpha=cfg["alpha"])
    # identity-fill unidentified (never-perturbed) rows with -e_r (decoupled self-decay prior) so
    # the nonlinear fixed-point solve is non-singular & stable -- the spike-1 identity-fill that
    # behaved best. Train-gene rows keep their estimate; the test doubles are pairs of train genes.
    trset = set(int(k) for k in train_genes)
    for r in range(G):
        if r not in trset:
            A_hat[r, :] = 0.0
            A_hat[r, r] = -1.0
    b_hat = estimate_b(A_hat, ctrl_mean, lam, s=s)

    coexpr = np.corrcoef(ctrl_cells, rowvar=False)
    tau_stack = np.stack([tau_obs[int(k)] for k in train_genes])
    corr_se = corr_single_effects(ctrl_cells, train_genes, mag)
    gears_se = gears_single_effects(tau_stack, train_genes, coexpr, knn=cfg["knn"])
    obs_se = {int(k): tau_obs[int(k)] for k in train_genes}
    return dict(A_hat=A_hat, b_hat=b_hat, corr=corr_se, gears=gears_se, obs=obs_se,
                true_single=true_single)


def run_instance(mode, seed, lam, cfg=CONFIG):
    """One (mode, seed, lambda) -> list of per-double records (empty list if dropped for instability)."""
    G, s = cfg["G"], cfg["s"]
    srng = np.random.default_rng(_skey(cfg["struct_base"], mode, seed))
    nrng = np.random.default_rng(_skey(cfg["noise_base"], mode, seed))
    (A_C, b_C), (A_Cp, b_Cp) = make_context_pair(G, cfg["n_reg"], mode, srng)
    train_genes = list(srng.permutation(G)[: cfg["P_train"]])
    mag = {int(k): float(srng.uniform(cfg["mag_lo"], cfg["mag_hi"])) for k in range(G)}
    gamma = {k: gamma_for_gene(k, G, srng, mag=mag[k]) for k in range(G)}
    all_pairs = list(itertools.combinations(train_genes, 2))
    idx = srng.choice(len(all_pairs), size=min(cfg["n_pairs"], len(all_pairs)), replace=False)
    pairs = [all_pairs[t] for t in idx]

    bC = _context_bundle(A_C, b_C, train_genes, gamma, mag, lam, cfg, nrng)
    bCp = _context_bundle(A_Cp, b_Cp, train_genes, gamma, mag, lam, cfg, nrng)
    if bC is None or bCp is None:
        return []  # dropped: unstable / non-convergent fixed points at this lambda

    rows = []
    for (i, j) in pairs:
        i, j = int(i), int(j)
        gij = gamma[i] + gamma[j]
        label, agree, ok = true_transport_label_nl(A_C, b_C, A_Cp, b_Cp, gij, lam,
                                                    thresh=cfg["thresh"], s=s)
        if not ok:
            continue
        # true epistasis magnitude in context C
        tij_true, okt = true_effect_nl(A_C, b_C, gij, lam, s=s)
        add_true = bC["true_single"][i] + bC["true_single"][j]
        epi = float(np.linalg.norm(tij_true - add_true)) if okt else np.nan
        epi_rel = epi / (np.linalg.norm(tij_true) + 1e-9) if okt else np.nan
        rows.append(dict(
            lam=lam, mode=mode, seed=seed, i=i, j=j, label=label, agree=agree,
            mechanism=transport_score_double(bC["A_hat"], bC["b_hat"], bCp["A_hat"], bCp["b_hat"],
                                             gamma[i], gamma[j], lam, s=s),
            corr_add=additive_null_double(bC["corr"], bCp["corr"], i, j),
            gears_add=additive_null_double(bC["gears"], bCp["gears"], i, j),
            obs_add=additive_null_double(bC["obs"], bCp["obs"], i, j),
            lin_oracle=linear_oracle_transport(A_C, A_Cp, gij),
            epi_abs=epi, epi_rel=epi_rel,
        ))
    return rows


def generate(lams, n_seeds, cfg=CONFIG, verbose=True):
    rows, drops = [], {}
    for lam in lams:
        d = 0
        for mode in MODES:
            for seed in range(n_seeds):
                r = run_instance(mode, seed, lam, cfg)
                if not r:
                    d += 1
                rows.extend(r)
        drops[lam] = d
        if verbose:
            print(f"  lam={lam:<4}: {d}/{len(MODES)*n_seeds} instances dropped (unstable)")
    return pd.DataFrame(rows), drops


# ---------------------------------------------------------------------------
# Metric: per-lambda pooled AUROC (or Spearman fallback) + cluster-bootstrap gap
# ---------------------------------------------------------------------------
def _pooled_metric(sub, col):
    """AUROC if both classes present, else Spearman(score, agree). Returns (value, kind)."""
    lab = sub["label"].to_numpy()
    if len(np.unique(lab)) == 2 and lab.sum() >= 3 and (len(lab) - lab.sum()) >= 3:
        return roc_auc_score(lab, sub[col].to_numpy()), "AUROC"
    rho = spearmanr(sub[col], sub["agree"]).correlation
    return (float(rho) if np.isfinite(rho) else np.nan), "Spearman"


def gap_curve(df, cfg=CONFIG):
    rng = np.random.default_rng(cfg["boot_seed"])
    out = []
    for lam, dl in df.groupby("lam"):
        row = {"lam": lam, "n": len(dl),
               "epi_abs": float(dl["epi_abs"].mean()), "epi_rel": float(dl["epi_rel"].mean()),
               "frac_transportable": float(dl["label"].mean())}
        kinds = set()
        for m in METHODS:
            v, kind = _pooled_metric(dl, m)
            row[m] = v
            kinds.add(kind)
        row["metric"] = "AUROC" if kinds == {"AUROC"} else "/".join(sorted(kinds))
        # cluster bootstrap on the mechanism - corr_add gap
        groups = {k: g for k, g in dl.groupby(["mode", "seed"])}
        keys = list(groups.keys())
        boot = []
        for _ in range(cfg["n_boot"]):
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            sb = pd.concat([groups[keys[t]] for t in pick], ignore_index=True)
            gm, _ = _pooled_metric(sb, "mechanism")
            gc, _ = _pooled_metric(sb, "corr_add")
            if np.isfinite(gm) and np.isfinite(gc):
                boot.append(gm - gc)
        boot = np.array(boot, float)
        row["gap"] = row["mechanism"] - row["corr_add"]
        row["gap_lo"] = float(np.percentile(boot, 2.5)) if len(boot) else np.nan
        row["gap_hi"] = float(np.percentile(boot, 97.5)) if len(boot) else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values("lam").reset_index(drop=True)


def plot_curve(curve, path):
    fig, ax1 = plt.subplots(figsize=(8.2, 5.6))
    ax1.axhline(0, color="black", lw=1, alpha=0.6)
    ax1.plot(curve["lam"], curve["gap"], "-o", color="#1f77b4", lw=2, label="mechanism − correlation gap")
    ax1.fill_between(curve["lam"], curve["gap_lo"], curve["gap_hi"], color="#1f77b4", alpha=0.18,
                     label="95% CI (cluster bootstrap)")
    ax1.set_xlabel("nonlinearity  λ"); ax1.set_ylabel("mechanism − correlation gap (pooled AUROC)")
    ax1.set_title("Spike #2 — does nonlinearity/epistasis let the mechanism overtake correlation?")
    ax1.grid(True, alpha=0.15)
    ax2 = ax1.twinx()
    ax2.plot(curve["lam"], curve["epi_rel"], "--s", color="#d62728", alpha=0.7, label="epistasis (relative)")
    ax2.set_ylabel("mean relative epistasis  ‖τ_ij − (τ_i+τ_j)‖ / ‖τ_ij‖", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    resdir = os.path.join(here, "results"); os.makedirs(resdir, exist_ok=True)
    cfg = CONFIG

    # ---- Kill probe: lam in {0, lam_high} ----
    print("== KILL PROBE ==  lam in {0, %.2f}, %d seeds/mode" % (cfg["lam_high"], cfg["n_seeds_probe"]))
    probe_df, probe_drops = generate([0.0, cfg["lam_high"]], cfg["n_seeds_probe"], cfg)
    pc = gap_curve(probe_df, cfg)
    g0 = pc[pc["lam"] == 0.0].iloc[0]; gh = pc[pc["lam"] == cfg["lam_high"]].iloc[0]
    print(pc[["lam", "metric", "mechanism", "corr_add", "gears_add", "obs_add", "lin_oracle",
              "epi_rel", "gap", "gap_lo", "gap_hi", "frac_transportable"]].to_string(index=False))

    anchor_ok = (g0["gap"] <= 0.02) and (g0["epi_rel"] < 0.01)          # M0
    moved = (gh["gap"] - g0["gap"] >= 0.10) or (gh["gap"] > 0)          # M1
    print(f"\nM0 anchor (lam=0): gap={g0['gap']:+.3f} (need <=~0), epi_rel={g0['epi_rel']:.4f} (need ~0) "
          f"-> {'OK' if anchor_ok else 'BROKEN'}")
    print(f"M1 movement: gap({cfg['lam_high']})={gh['gap']:+.3f} vs gap(0)={g0['gap']:+.3f}; "
          f"delta={gh['gap']-g0['gap']:+.3f} -> {'MOVED' if moved else 'FLAT'}")

    if not anchor_ok:
        print("\nANCHOR BROKEN -- stopping. Fix before proceeding (see FINDINGS).")
        df, curve = probe_df, pc
    elif not moved:
        print("\nDECISION: PARK IT. Nonlinearity did not erode correlation's edge with this estimator.")
        df, curve = probe_df, pc
    else:
        print("\nDECISION: gap moved -> running the full lambda grid.")
        grid_df, grid_drops = generate(cfg["lam_grid"], cfg["n_seeds_full"], cfg)
        df, curve = grid_df, gap_curve(grid_df, cfg)

    df.to_csv(os.path.join(resdir, "spike2_records.csv"), index=False)
    curve.to_csv(os.path.join(resdir, "gap_vs_lambda.csv"), index=False)
    plot_curve(curve, os.path.join(resdir, "gap_vs_lambda.png"))
    print(f"\nWrote results/spike2_records.csv, results/gap_vs_lambda.csv, results/gap_vs_lambda.png")


if __name__ == "__main__":
    main()
