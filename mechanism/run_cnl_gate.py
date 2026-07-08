"""C-NL gate harness -- does the baseline third moment predict what covariance cannot?

Pipeline (symmetric A, genuine nonlinear-SDE sampling; see causaldgp / response):
  - control latent samples -> baseline covariance Sigma and third moment T
  - symmetric +/-m single-gene perturbations -> pure second-order response
      c_ik = [Delta_mu_i(+m e_k) + Delta_mu_i(-m e_k)] / (2 m^2)
    (cancels the linear part and any chi!=Sigma mismatch exactly; = 0 at lambda=0 by construction)
  - Test 1: does c (2nd-order response) and T grow with lambda?
  - Test 2: does the third-moment feature T[e_k,e_k]_i = T_ikk predict c better than the covariance
            surrogate (Sigma e_k)_i^2 = Sigma_ik^2 ? (coefficient FIT, not assumed -- provenance rule)
  - Test 3: re-estimate T at reduced depth (latent subsample + NB emission on) -> estimability threshold

PROVENANCE: the third-moment link is an INFERENCE from fluctuation-response theory, not a CIPHER
result. T[u,u] is a candidate predictor with a fitted coefficient. See FINDINGS_CNL.md section 0.

Run: python run_cnl_gate.py   -> results/delta_r2_vs_lambda.{csv,png}, results/depth_threshold.{csv,png}
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causaldgp import (make_A_symmetric, em_sample_latent, em_perturb_responses, emit_from_latent)
from response import (empirical_third_moment, second_order_term, covariance_surrogate, r2_fit)

CONFIG = dict(
    G=50, n_reg=6, m=2.5, P=20, b_scale=1.0,   # b~N(0,b_scale): nonzero baseline moves the operating
                                               # point off x*=0, else the odd tanh gives a symmetric
                                               # stationary law with ZERO third moment (nothing to test)
    n_ctrl=100_000, n_pert=5_000, dt=0.02, n_steps=500,
    lam_grid=[0.0, 0.25, 0.5, 0.7, 0.85],
    n_seeds=8,
    depths=[100_000, 30_000, 10_000, 3_000, 1_000],
    depth_lam=0.85,               # lambda at which the depth threshold is characterized (strongest signal)
    seed_base=500, n_boot=2000, boot_seed=321,
)


def run_instance(lam, seed, cfg=CONFIG, with_depth=False):
    G, m, P = cfg["G"], cfg["m"], cfg["P"]
    srng = np.random.default_rng(cfg["seed_base"] + seed)                 # structure (lambda-independent -> paired)
    nrng = np.random.default_rng(cfg["seed_base"] + 10_000 + seed)        # sim noise
    A = make_A_symmetric(G, cfg["n_reg"], srng)
    b = srng.normal(0.0, cfg["b_scale"], G)   # nonzero -> operating point off 0 -> genuine non-Gaussianity
    genes = [int(k) for k in srng.permutation(G)[:P]]

    ctrl = em_sample_latent(A, b, np.zeros(G), lam, cfg["n_ctrl"], nrng, dt=cfg["dt"], n_steps=cfg["n_steps"])
    Sig = np.cov(ctrl, rowvar=False)
    T = empirical_third_moment(ctrl)

    U = np.zeros((2 * P, G))
    for kk, k in enumerate(genes):
        U[2 * kk, k] = +m
        U[2 * kk + 1, k] = -m
    dmu = em_perturb_responses(A, b, U, lam, cfg["n_pert"], nrng, dt=cfg["dt"], n_steps=cfg["n_steps"])

    C, F, B, eks = [], [], [], []
    for kk, k in enumerate(genes):
        c = (dmu[2 * kk] + dmu[2 * kk + 1]) / (2 * m * m)                 # pure 2nd-order response (G,)
        ek = np.zeros(G); ek[k] = 1.0
        eks.append(ek)
        C.append(c); F.append(second_order_term(T, ek)); B.append(covariance_surrogate(Sig, ek))
    C = np.concatenate(C); F = np.concatenate(F); B = np.concatenate(B)

    out = dict(lam=lam, seed=seed, C=C, F=F, B=B,
               c_norm=float(np.sqrt((C ** 2).mean())), T_norm=float(np.sqrt((T ** 2).mean())))

    if with_depth:
        emitted = np.log1p(emit_from_latent(ctrl, nrng))                  # realistic observation (log NB counts)
        df = {}
        for d in cfg["depths"]:
            if d > cfg["n_ctrl"]:
                continue
            idx = nrng.choice(cfg["n_ctrl"], d, replace=False)
            Tlat = empirical_third_moment(ctrl[idx])
            Temit = empirical_third_moment(emitted[idx])
            df[d] = dict(latent=np.concatenate([second_order_term(Tlat, ek) for ek in eks]),
                         emit=np.concatenate([second_order_term(Temit, ek) for ek in eks]))
        out["depth_feats"] = df
    return out


# ---------------------------------------------------------------------------
def _boot_delta_r2(insts, feat_key="F", rng=None, n_boot=2000):
    """Point estimate + 95% CI of dR2 = R2(C~feat) - R2(C~B), cluster-bootstrapping over seeds."""
    C = np.concatenate([d["C"] for d in insts])
    Ff = np.concatenate([d[feat_key] for d in insts])
    Bb = np.concatenate([d["B"] for d in insts])
    point = r2_fit(C, Ff) - r2_fit(C, Bb)
    rt, rc = r2_fit(C, Ff), r2_fit(C, Bb)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(insts), len(insts))
        c = np.concatenate([insts[i]["C"] for i in pick])
        f = np.concatenate([insts[i][feat_key] for i in pick])
        bb = np.concatenate([insts[i]["B"] for i in pick])
        boots.append((r2_fit(c, f) - r2_fit(c, bb)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(dr2=point, r2_T=rt, r2_cov=rc, lo=float(lo), hi=float(hi))


def main(quick=False):
    here = os.path.dirname(os.path.abspath(__file__))
    resdir = os.path.join(here, "results"); os.makedirs(resdir, exist_ok=True)
    cfg = dict(CONFIG)
    if quick:  # fast directional read (not the committed artifact)
        cfg.update(n_ctrl=40_000, n_pert=4_000, n_seeds=5, n_boot=800,
                   depths=[40_000, 10_000, 3_000, 1_000], depth_lam=0.85)
        print("[QUICK MODE] reduced settings for a directional read\n")
    rng = np.random.default_rng(cfg["boot_seed"])

    # ---- Tests 1 + 2: lambda sweep ----
    print("== C-NL gate: lambda sweep (Tests 1 + 2) ==")
    rows = []
    for lam in cfg["lam_grid"]:
        insts = [run_instance(lam, s, cfg) for s in range(cfg["n_seeds"])]
        bd = _boot_delta_r2(insts, "F", rng, cfg["n_boot"])
        row = dict(lam=lam,
                   c_norm=float(np.mean([d["c_norm"] for d in insts])),
                   T_norm=float(np.mean([d["T_norm"] for d in insts])),
                   r2_T=round(bd["r2_T"], 4), r2_cov=round(bd["r2_cov"], 4),
                   delta_r2=round(bd["dr2"], 4), ci_lo=round(bd["lo"], 4), ci_hi=round(bd["hi"], 4))
        rows.append(row)
        star = "  <-- CI excludes 0" if bd["lo"] > 0 else ""
        print(f"  lam={lam:<5} c_norm={row['c_norm']:.4f} T_norm={row['T_norm']:.4f} | "
              f"R2_T={row['r2_T']:.3f} R2_cov={row['r2_cov']:.3f} dR2={row['delta_r2']:+.3f} "
              f"[{row['ci_lo']:+.3f},{row['ci_hi']:+.3f}]{star}")
    curve = pd.DataFrame(rows)
    curve.to_csv(os.path.join(resdir, "delta_r2_vs_lambda.csv"), index=False)

    # ---- Test 3: depth sweep at depth_lam ----
    print(f"\n== C-NL gate: depth sweep (Test 3) at lambda={cfg['depth_lam']} ==")
    dinsts = [run_instance(cfg["depth_lam"], s, cfg, with_depth=True) for s in range(cfg["n_seeds"])]
    drows = []
    for d in cfg["depths"]:
        if d > cfg["n_ctrl"]:
            continue
        lat = [dict(C=x["C"], F=x["depth_feats"][d]["latent"], B=x["B"]) for x in dinsts]
        emi = [dict(C=x["C"], F=x["depth_feats"][d]["emit"], B=x["B"]) for x in dinsts]
        bl = _boot_delta_r2(lat, "F", rng, cfg["n_boot"])
        be = _boot_delta_r2(emi, "F", rng, cfg["n_boot"])
        drows.append(dict(depth=d, r2_T_latent=round(bl["r2_T"], 4), dr2_latent=round(bl["dr2"], 4),
                          ci_lo_latent=round(bl["lo"], 4), ci_hi_latent=round(bl["hi"], 4),
                          r2_T_emit=round(be["r2_T"], 4), dr2_emit=round(be["dr2"], 4),
                          ci_lo_emit=round(be["lo"], 4), ci_hi_emit=round(be["hi"], 4)))
        print(f"  depth={d:>7} | latent dR2={bl['dr2']:+.3f}[{bl['lo']:+.3f},{bl['hi']:+.3f}]  "
              f"emit dR2={be['dr2']:+.3f}[{be['lo']:+.3f},{be['hi']:+.3f}]")
    depth = pd.DataFrame(drows)
    depth.to_csv(os.path.join(resdir, "depth_threshold.csv"), index=False)

    _plot(curve, depth, cfg, resdir)
    print("\nWrote results/delta_r2_vs_lambda.{csv,png}, results/depth_threshold.{csv,png}")


def _plot(curve, depth, cfg, resdir):
    # dR2 vs lambda
    fig, ax1 = plt.subplots(figsize=(7.8, 5.4))
    ax1.axhline(0, color="black", lw=1, alpha=0.6)
    ax1.plot(curve["lam"], curve["delta_r2"], "-o", color="#1f77b4", lw=2, label="ΔR² (T[u,u] − covariance surrogate)")
    ax1.fill_between(curve["lam"], curve["ci_lo"], curve["ci_hi"], color="#1f77b4", alpha=0.18, label="95% CI")
    ax1.set_xlabel("nonlinearity  λ"); ax1.set_ylabel("ΔR²  (third moment beats covariance)")
    ax1.set_title("C-NL gate — does the baseline third moment predict the 2nd-order response?")
    ax1.grid(True, alpha=0.15)
    ax2 = ax1.twinx()
    ax2.plot(curve["lam"], curve["T_norm"], "--s", color="#d62728", alpha=0.7, label="‖T‖ (baseline third moment)")
    ax2.plot(curve["lam"], curve["c_norm"], ":^", color="#2ca02c", alpha=0.7, label="‖c‖ (2nd-order response)")
    ax2.set_ylabel("‖T‖  and  ‖c‖", color="#555")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(resdir, "delta_r2_vs_lambda.png"), dpi=150); plt.close(fig)

    # depth threshold
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.axhline(0, color="black", lw=1, alpha=0.6)
    ax.plot(depth["depth"], depth["dr2_latent"], "-o", color="#1f77b4", label="ΔR² latent (pure T-estimation variance)")
    ax.fill_between(depth["depth"], depth["ci_lo_latent"], depth["ci_hi_latent"], color="#1f77b4", alpha=0.15)
    ax.plot(depth["depth"], depth["dr2_emit"], "-s", color="#ff7f0e", label="ΔR² emission-on (realistic observation)")
    ax.fill_between(depth["depth"], depth["ci_lo_emit"], depth["ci_hi_emit"], color="#ff7f0e", alpha=0.15)
    ax.set_xscale("log"); ax.set_xlabel("control cells (depth)"); ax.set_ylabel("ΔR²")
    ax.set_title(f"C-NL gate — Test 3: third-moment estimability vs depth (λ={cfg['depth_lam']})")
    ax.grid(True, alpha=0.15); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(resdir, "depth_threshold.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    import sys
    main(quick=("quick" in sys.argv))
