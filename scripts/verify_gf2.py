#!/usr/bin/env python
"""Adversarial verification of the G-F.2 result before it is written into the gate.

Stress-tests the three things that could overturn "causal does NOT beat twin on external edges":
  (1) SIGN CONVENTION — recompute under every convention (derived / all-flipped / all-unflipped /
      swapped). If ANY convention makes causal beat twin, the derived one may be wrong.
  (2) EDGE INDEPENDENCE — the 6122 edges cluster into 9 regulators; edges within a regulator are not
      independent. Regulator-level sign test + cluster bootstrap (resample REGULATORS) give the honest CI.
  (3) CAUSAL vs TWIN — paired McNemar on pooled edges + regulator-level paired comparison: is
      causal-twin a real difference or within noise?
Read-only on the committed prediction parquets + edge table.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load():
    c = pd.read_parquet(os.path.join(RES, "fusion_pred_causal.parquet"))
    n = pd.read_parquet(os.path.join(RES, "fusion_pred_noncausal.parquet"))
    e = pd.read_csv(os.path.join(RES, "fusion_measurable_edges.csv"))
    cmap = {g: i for i, g in enumerate(c.columns)}
    rmap = {r: i for i, r in enumerate(c.index)}
    e = e[e.reg_ens.isin(rmap) & e.tgt_ens.isin(cmap)].copy()
    return c.to_numpy(float), n.to_numpy(float), rmap, cmap, e


def signs(P, e, rmap, cmap):
    ri = e.reg_ens.map(rmap).to_numpy(); ci = e.tgt_ens.map(cmap).to_numpy()
    return np.sign(P[ri, ci])


def expected(e, mode):
    s = e.sign.values
    fre = e.src.values == "freimer"
    if mode == "derived":     # Freimer KO logFC direct; Weinstock beta flips under knockdown
        return np.where(fre, s, -s)
    if mode == "all_flipped":
        return -s
    if mode == "all_unflipped":
        return s
    if mode == "swapped":     # deliberately wrong: flip Freimer, keep Weinstock
        return np.where(fre, -s, s)


def acc(sg, exp):
    return float((sg == exp).mean())


def null_p(P, e, rmap, cmap, exp, B=20000, seed=0):
    ri = e.reg_ens.map(rmap).to_numpy()
    rng = np.random.default_rng(seed); nH = P.shape[1]
    obs = acc(np.sign(P[ri, e.tgt_ens.map(cmap).to_numpy()]), exp)
    nul = np.array([acc(np.sign(P[ri, rng.integers(0, nH, len(ri))]), exp) for _ in range(B)])
    return obs, float(nul.mean()), float(nul.std()), (np.sum(nul >= obs) + 1) / (B + 1)


def mcnemar(sc, sn, exp):
    cc = sc == exp; nn = sn == exp
    b = int(np.sum(cc & ~nn)); c = int(np.sum(~cc & nn))
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from math import erfc, sqrt
    p = erfc(sqrt(chi2 / 2))   # chi2_1 -> normal tail
    return b, c, p


def main():
    C, N, rmap, cmap, e = load()
    print(f"[data] held-out regs predicted={len(rmap)}  measurable edges={len(e)} "
          f"(weinstock={int((e.src=='weinstock').sum())} freimer={int((e.src=='freimer').sum())})")

    # (1) sign-convention robustness — does any convention make causal beat twin?
    print("\n=== (1) SIGN-CONVENTION ROBUSTNESS (combined edges) ===")
    print(f"{'convention':14s} {'causal':>7s} {'twin':>7s} {'c-t':>7s}")
    for mode in ("derived", "all_flipped", "all_unflipped", "swapped"):
        exp = expected(e, mode)
        ac = acc(signs(C, e, rmap, cmap), exp); an = acc(signs(N, e, rmap, cmap), exp)
        print(f"{mode:14s} {ac:7.3f} {an:7.3f} {ac-an:+7.3f}")

    # main table, derived convention, per source
    print("\n=== (2)+(3) DERIVED CONVENTION — per source ===")
    for src, sub in [("weinstock", e[e.src == "weinstock"]), ("freimer", e[e.src == "freimer"]), ("combined", e)]:
        if len(sub) == 0:
            continue
        exp = expected(sub, "derived")
        oc, mnc, sdc, pc = null_p(C, sub, rmap, cmap, exp)
        on, mnn, sdn, pn = null_p(N, sub, rmap, cmap, exp)
        b, cc_, pm = mcnemar(signs(C, sub, rmap, cmap), signs(N, sub, rmap, cmap), exp)
        print(f"[{src}] n={len(sub)} regs={sub.reg_ens.nunique()} | causal={oc:.3f}(p_null={pc:.1e}) "
              f"twin={on:.3f}(p_null={pn:.1e}) | c-t={oc-on:+.3f} McNemar b={b} c={cc_} p={pm:.3f}")

    # (2) regulator-level (the honest unit): per-reg causal vs twin, sign test across regs
    print("\n=== (2) REGULATOR-LEVEL (edges within a regulator are not independent) ===")
    rows = []
    for r, sub in e.groupby("reg_ens"):
        if len(sub) < 5:   # drop n<5 (TBX21 n=1)
            continue
        exp = expected(sub, "derived")
        ac = acc(signs(C, sub, rmap, cmap), exp); an = acc(signs(N, sub, rmap, cmap), exp)
        rows.append((sub.reg_sym.iloc[0], len(sub), ac, an, ac - an))
    df = pd.DataFrame(rows, columns=["reg", "n", "causal", "twin", "c_minus_t"]).sort_values("n", ascending=False)
    print(df.to_string(index=False))
    n_reg = len(df)
    n_c_gt_half = int((df.causal > 0.5).sum())
    n_c_gt_twin = int((df.c_minus_t > 0).sum())
    # binomial two-sided p for k/n at 0.5
    from math import comb
    def binom_p(k, n):
        return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    print(f"\nregulators with causal>0.5: {n_c_gt_half}/{n_reg}  (binom p={binom_p(max(n_c_gt_half, n_reg-n_c_gt_half), n_reg):.3f})")
    print(f"regulators with causal>twin: {n_c_gt_twin}/{n_reg}  (binom p={binom_p(max(n_c_gt_twin, n_reg-n_c_gt_twin), n_reg):.3f})")

    # (2) cluster bootstrap over regulators — CI on causal acc, twin acc, causal-twin
    print("\n=== (2) CLUSTER BOOTSTRAP over regulators (B=10000) ===")
    regs = df.reg.tolist()
    per = {r: e[e.reg_sym == r] for r in regs}
    rng = np.random.default_rng(0); B = 10000
    ca, ta, dt = [], [], []
    for _ in range(B):
        pick = rng.choice(regs, size=len(regs), replace=True)
        sub = pd.concat([per[r] for r in pick])
        exp = expected(sub, "derived")
        a = acc(signs(C, sub, rmap, cmap), exp); b = acc(signs(N, sub, rmap, cmap), exp)
        ca.append(a); ta.append(b); dt.append(a - b)
    def ci(x): return np.percentile(x, [2.5, 97.5])
    print(f"causal acc   : {np.mean(ca):.3f}  95%CI [{ci(ca)[0]:.3f}, {ci(ca)[1]:.3f}]")
    print(f"twin acc     : {np.mean(ta):.3f}  95%CI [{ci(ta)[0]:.3f}, {ci(ta)[1]:.3f}]")
    print(f"causal-twin  : {np.mean(dt):+.3f} 95%CI [{ci(dt)[0]:+.3f}, {ci(dt)[1]:+.3f}]  "
          f"(excludes 0: {not (ci(dt)[0] <= 0 <= ci(dt)[1])})")
    print(f"causal vs 0.5: CI excludes 0.5: {not (ci(ca)[0] <= 0.5 <= ci(ca)[1])}")
    print("\nVERIFY_DONE")


if __name__ == "__main__":
    main()
