#!/usr/bin/env python
"""C-FUSE G-F.2 (powered, non-circular) — does the re-trained do-operator recover the DIRECTION of
held-out external causal edges better than its non-causal twin and a degree-preserving null?

Inputs (all produced by scripts/fusion_retrain.py, which held the external regulators OUT of training):
  results/fusion_pred_causal.parquet     Delta_hat[R, T] over HVG for each held-out regulator R (do-operator)
  results/fusion_pred_noncausal.parquet  same, non-causal twin
  results/fusion_measurable_edges.csv    external edges (reg_ens, tgt_ens, sign, src) — Weinstock + Freimer

Sign convention (critical):
  Our prediction = Delta under CRISPRi KNOCKDOWN of R (perturbed - control).
  * Freimer edges: `sign` = sign(logFC) of the R-KO experiment = same knockdown convention.
        expected sign(Delta_pred) = edge.sign
  * Weinstock edges: `sign` = sign(beta), the LLCB activation coefficient of R->T (presence/activity).
        Knocking R down flips it: expected sign(Delta_pred) = -edge.sign
We report accuracy under this derived convention (and the raw, unflipped accuracy, so a convention error
would be visible).

Test: per source (Weinstock direct / Freimer DE) and combined —
  observed sign-accuracy (causal, twin) vs a DEGREE-PRESERVING null (each edge keeps its regulator, its
  target is resampled uniformly over the HVG panel; preserves per-regulator degree + the model's marginal
  sign bias). p = fraction of null accuracy >= observed. PASS: causal > twin AND p_causal < 0.01.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
B = 20000       # null permutations
SEED = 0


def _acc_and_null(P, ri, ci, expected, nH, rng):
    """observed sign-accuracy + degree-preserving null distribution for a prediction matrix P."""
    obs = float((np.sign(P[ri, ci]) == expected).mean())
    null = np.empty(B, dtype=float)
    ne = len(ri)
    for b in range(B):
        cj = rng.integers(0, nH, size=ne)
        null[b] = (np.sign(P[ri, cj]) == expected).mean()
    p = (np.sum(null >= obs) + 1) / (B + 1)
    return obs, float(null.mean()), float(null.std()), float(p)


def _eval(edges, preds, label, rows, rng):
    """Evaluate one edge subset across causal + twin, print + collect a result row."""
    Pc, rmapc, cmap = preds["causal"]
    Pn, rmapn, _ = preds["noncausal"]
    e = edges[edges.reg_ens.isin(rmapc) & edges.tgt_ens.isin(cmap)].copy()
    if len(e) == 0:
        print(f"[{label}] no measurable held-out edges — skipped")
        return
    expected = np.where(e.src.values == "freimer", e.sign.values, -e.sign.values).astype(int)
    ci = e.tgt_ens.map(cmap).to_numpy()
    ri_c = e.reg_ens.map(rmapc).to_numpy()
    ri_n = e.reg_ens.map(rmapn).to_numpy()
    nH = Pc.shape[1]
    oc, mnc, sdc, pc = _acc_and_null(Pc, ri_c, ci, expected, nH, np.random.default_rng(SEED))
    on, mnn, sdn, pn = _acc_and_null(Pn, ri_n, ci, expected, nH, np.random.default_rng(SEED))
    raw_c = float((np.sign(Pc[ri_c, ci]) == e.sign.values).mean())   # unflipped, sanity
    verdict = "PASS" if (oc > on and pc < 0.01) else "FAIL"
    print(f"[{label}] edges={len(e)} regs={e.reg_ens.nunique()} | "
          f"causal acc={oc:.3f} (null {mnc:.3f}±{sdc:.3f}, p={pc:.2e}) | "
          f"twin acc={on:.3f} (p={pn:.2e}) | causal-twin={oc-on:+.3f} | {verdict}")
    rows.append(dict(subset=label, edges=len(e), regulators=int(e.reg_ens.nunique()),
                     causal_acc=round(oc, 4), twin_acc=round(on, 4), causal_minus_twin=round(oc - on, 4),
                     null_mean=round(mnc, 4), null_sd=round(sdc, 4),
                     p_causal_vs_null=pc, p_twin_vs_null=pn, raw_causal_acc=round(raw_c, 4),
                     verdict=verdict))


def _load_pred(name):
    df = pd.read_parquet(os.path.join(RES, f"fusion_pred_{name}.parquet"))
    P = df.to_numpy(dtype=float)
    rmap = {r: i for i, r in enumerate(df.index)}
    cmap = {c: i for i, c in enumerate(df.columns)}
    return P, rmap, cmap


def main():
    preds = {n: _load_pred(n) for n in ("causal", "noncausal")}
    tested = set(preds["causal"][1])
    edges = pd.read_csv(os.path.join(RES, "fusion_measurable_edges.csv"))
    held = edges[edges.reg_ens.isin(tested)].copy()
    print(f"[gf2] held-out regulators predicted={len(tested)}  measurable held-out edges={len(held)} "
          f"(weinstock={int((held.src=='weinstock').sum())} freimer={int((held.src=='freimer').sum())})")
    print(f"[gf2] regulators: {sorted(tested)}")

    rows = []
    rng = np.random.default_rng(SEED)
    _eval(held[held.src == "weinstock"], preds, "weinstock(direct)", rows, rng)
    _eval(held[held.src == "freimer"], preds, "freimer(DE)", rows, rng)
    _eval(held, preds, "combined", rows, rng)

    # per-regulator breakdown (causal, derived convention)
    Pc, rmapc, cmap = preds["causal"]
    perreg = []
    for r, sub in held.groupby("reg_ens"):
        sub = sub[sub.tgt_ens.isin(cmap)]
        if len(sub) == 0:
            continue
        exp = np.where(sub.src.values == "freimer", sub.sign.values, -sub.sign.values)
        ci = sub.tgt_ens.map(cmap).to_numpy()
        acc = float((np.sign(Pc[rmapc[r], ci]) == exp).mean())
        perreg.append(dict(reg_ens=r, reg_sym=sub.reg_sym.iloc[0], n_edges=len(sub), causal_acc=round(acc, 4)))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "fusion_gf2.csv"), index=False)
    pd.DataFrame(perreg).sort_values("n_edges", ascending=False).to_csv(
        os.path.join(RES, "fusion_gf2_perreg.csv"), index=False)
    print("\n########## G-F.2 RESULT ##########")
    print(out.to_string(index=False))
    comb = out[out.subset == "combined"]
    if len(comb):
        v = comb.iloc[0]
        print(f"\nHEADLINE (combined): causal={v.causal_acc} twin={v.twin_acc} "
              f"p_causal_vs_null={v.p_causal_vs_null:.2e} -> {v.verdict}")
    print("GF2_DONE", flush=True)


if __name__ == "__main__":
    main()
