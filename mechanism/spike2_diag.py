"""Spike #2 supplementary diagnostics -- the 'why' behind the flat gap.

(1) Finer continuous metric: Spearman(score, agree) per lambda, to rule out that the near-1.0
    binary-AUROC ceiling hides a mechanism advantage.
(2) Effect capture: mean cos(predicted double effect, TRUE double effect) per method per lambda --
    does the mechanism actually predict the epistatic double better than simply summing observed
    singles? Writes results/spike2_diagnostics.csv.
"""

from __future__ import annotations
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from causaldgp import make_context_pair, gamma_for_gene, true_effect_nl
from mechanism import predict_double_mech
from run_spike2 import CONFIG, MODES, _skey, _context_bundle

LAMS = [0.0, 0.25, 0.5, 0.7, 0.85]


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = CONFIG; G, s = cfg["G"], cfg["s"]

    # (1) Spearman on the full-grid records
    recs = pd.read_csv(os.path.join(here, "results", "spike2_records.csv"))

    rows = []
    for lam in LAMS:
        dl = recs[np.isclose(recs["lam"], lam)]
        def sp(c):
            if len(dl) == 0:
                return np.nan
            r = spearmanr(dl[c], dl["agree"]).correlation
            return float(r) if np.isfinite(r) else np.nan

        # (2) effect capture on full seed set
        mech, obs, corr, epis = [], [], [], []
        for mode in MODES:
            for seed in range(cfg["n_seeds_full"]):
                srng = np.random.default_rng(_skey(cfg["struct_base"], mode, seed))
                nrng = np.random.default_rng(_skey(cfg["noise_base"], mode, seed))
                (A_C, b_C), _ = make_context_pair(G, cfg["n_reg"], mode, srng)
                tg = list(srng.permutation(G)[: cfg["P_train"]])
                mag = {int(k): float(srng.uniform(cfg["mag_lo"], cfg["mag_hi"])) for k in range(G)}
                gam = {k: gamma_for_gene(k, G, srng, mag=mag[k]) for k in range(G)}
                allp = list(itertools.combinations(tg, 2))
                pairs = [allp[t] for t in srng.choice(len(allp), cfg["n_pairs"], replace=False)]
                bC = _context_bundle(A_C, b_C, tg, gam, mag, lam, cfg, nrng)
                if bC is None:
                    continue
                for (i, j) in pairs:
                    i, j = int(i), int(j)
                    tij, ok = true_effect_nl(A_C, b_C, gam[i] + gam[j], lam, s=s)
                    if not ok:
                        continue
                    pm = predict_double_mech(bC["A_hat"], bC["b_hat"], gam[i], gam[j], lam, s=s)
                    mech.append(_cos(pm, tij))
                    obs.append(_cos(bC["obs"][i] + bC["obs"][j], tij))
                    corr.append(_cos(bC["corr"][i] + bC["corr"][j], tij))
                    add_true = bC["true_single"][i] + bC["true_single"][j]
                    epis.append(np.linalg.norm(tij - add_true) / (np.linalg.norm(tij) + 1e-9))

        rows.append(dict(
            lam=lam,
            spearman_mech=round(sp("mechanism"), 4), spearman_corr=round(sp("corr_add"), 4),
            spearman_obs=round(sp("obs_add"), 4),
            spearman_gap=round(sp("mechanism") - sp("corr_add"), 4),
            effcos_mech=round(float(np.mean(mech)), 4),
            effcos_obs_add=round(float(np.mean(obs)), 4),
            effcos_corr_add=round(float(np.mean(corr)), 4),
            epi_rel=round(float(np.mean(epis)), 4),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(here, "results", "spike2_diagnostics.csv"), index=False)
    print(out.to_string(index=False))
    print("\nWrote results/spike2_diagnostics.csv")


if __name__ == "__main__":
    main()
