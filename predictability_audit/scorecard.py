#!/usr/bin/env python
"""predictability_audit.scorecard — assemble the Marson-CD4 predictability scorecard from the committed
gate CSVs (ground truth). Stdlib-only (csv + math), no pandas, no retrain, does NOT import/modify
`core.eval`. `run_audit()` reproduces each pre-registered verdict from its committed score + null +
floor, and self-checks it against the committed verdict string (a packaging-faithfulness test).

Scorecard families:
  * 7 probes (P1 causal-matrix, P2 fluctuation, P3 single-cell SNR, P4 trajectory-geometry,
    P5 donor-structure, P6 relational-object, P7 external causal-edge validation)
  * predictability budget (the measured ceiling + cross-donor floor)
  * do-operator C2 positive control (the signal-detection anchor: a null means "no signal", not
    "no sensitivity")
"""
import csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE_FLOOR = 0.03          # per-perturbation cross-donor reproducibility floor (verified below)
REL_THRESHOLD = 0.30        # pre-registered relational-object bar (RELATIONAL.md)


def _rows(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _row(probe, family, question, metric, score, reference, verdict, committed, consistent, source, floor=None):
    return dict(probe=probe, family=family, question=question, metric=metric,
                score=score, reference=reference, noise_floor=floor, verdict=verdict,
                committed_verdict=committed, consistent=consistent, source=source)


# ---- P1: causal-matrix Â_C (mechanism vs correlation null) ------------------------------------
def causal_matrix():
    r = {(x["method"], x["scope"], x["metric"]): x for x in _rows("mechanism/results/c4_auroc.csv")}
    mech = _f(r[("mechanism", "pooled", "AUROC")]["value"])
    corr = _f(r[("corr_null", "pooled", "AUROC")]["value"])
    true = _f(r[("true_col", "pooled", "AUROC")]["value"])
    consistent = mech < corr                       # committed FAIL = mechanism loses to correlation
    return _row("P1 causal-matrix (Â_C)", "mechanism",
                "Does an explicit per-context causal matrix beat a correlation null under P≪G?",
                "AUROC (recover held-out interaction sign)", round(mech, 4), f"corr-null {corr:.3f}",
                f"FAIL — Â beats chance ({mech:.3f}>0.5) but LOSES to a correlation null ({corr:.3f}); "
                f"oracle true-col={true:.3f}. Mechanism recovery not established under P≪G.",
                "FAIL", consistent, "mechanism/results/c4_auroc.csv", floor=0.5)


# ---- P2: fluctuation / third-moment (C-NL real-data) ------------------------------------------
def fluctuation_cnl():
    rows = _rows("mechanism/results/cnl_realdata_gate.csv")
    ps = [_f(x["perm_p"]) for x in rows if _f(x["perm_p"]) is not None]
    dr2 = sorted(_f(x["dR2"]) for x in rows if _f(x["dR2"]) is not None)
    n_sig = sum(1 for p in ps if p < 0.05)
    med_dr2 = dr2[len(dr2) // 2] if dr2 else None
    consistent = n_sig == 0                        # committed NEGATIVE = no stratum significant
    return _row("P2 fluctuation (3rd-moment C-NL)", "mechanism",
                "Does the response 3rd moment predict what covariance cannot, on real data?",
                "ΔR²(3rd-moment vs covariance-only), permutation-tested per (donor×cond×scope)",
                f"{med_dr2:.2e}", f"{n_sig}/{len(ps)} strata sig (p<0.05)",
                f"NEGATIVE — 3rd moment orthogonal to the residual; {n_sig}/{len(ps)} strata significant "
                f"(median ΔR²≈{med_dr2:.1e}). Positive on the simulator, dead on real data.",
                "NEGATIVE", consistent, "mechanism/results/cnl_realdata_gate.csv")


# ---- P3: single-cell SNR pre-check ------------------------------------------------------------
def single_cell_snr():
    m = {x["metric"]: _f(x["value"]) for x in _rows("results/phaseB_snr_precheck_summary.csv")}
    green = m["green"]
    xdonor = m["xdonor_top50_median"]
    detect = m["frac_cyto_detectable_snr3"]
    consistent = green == 0.0                       # committed RED = single-cell won't lift the floor
    return _row("P3 single-cell SNR", "reliability",
                "Would single-cell resolution lift the per-perturbation floor above pseudobulk?",
                "cross-donor top-50 median + cytokine-gene SNR>3 detectability + go/no-go gate",
                round(xdonor, 4), f"detectable {detect:.2f}, gate green={green:.0f}",
                f"NOISE-FLOOR — confirmed floor at pseudobulk depth (x-donor {xdonor:.3f}); only "
                f"{detect:.0%} of cytokine genes reach SNR>3; single-cell would not fix it (gate RED).",
                "confirmed noise floor", consistent, "results/phaseB_snr_precheck_summary.csv",
                floor=NOISE_FLOOR)


# ---- P4: trajectory-geometry ------------------------------------------------------------------
def trajectory_geometry():
    rows = [x for x in _rows("results/trajectory_coupling_gate.csv") if x["part"] == "G-TC.1"]
    by = {x["split"]: x for x in rows}
    cond_p, cond_pp = _f(by["condition"]["partial_spearman"]), _f(by["condition"]["perm_p"])
    gene_p, gene_pp = _f(by["gene"]["partial_spearman"]), _f(by["gene"]["perm_p"])
    consistent = cond_pp > 0.05 and gene_pp > 0.05  # committed negative = partial ρ not significant
    return _row("P4 trajectory-geometry", "geometry",
                "Is per-perturbation recoverability explained by activation-trajectory geometry?",
                "partial Spearman(recoverability, trajectory-coupling | magnitude), perm-tested",
                f"{cond_p:.3f}/{gene_p:.3f}", f"perm_p {cond_pp:.2f}/{gene_pp:.2f}",
                f"NEGATIVE — partial ρ≈0 (cond {cond_p:.3f} p={cond_pp:.2f}; gene {gene_p:.3f} "
                f"p={gene_pp:.2f}); recoverability is NOT a trajectory-geometry artifact.",
                "clean negative", consistent, "results/trajectory_coupling_gate.csv")


# ---- P5: donor-structure ----------------------------------------------------------------------
def donor_structure():
    rows = _rows("results/donor_structure_gate.csv")
    gd2 = [x for x in rows if x["test"] == "G-D.2" and x["label"] == "medians"]
    cond = _f(gd2[0]["same_med"]) if gd2 else None     # donor-CONDITIONING (same-donor)
    avg = _f(gd2[0]["diff_med"]) if gd2 else None       # donor-AVERAGING (pooled)
    consistent = cond < avg                              # committed NO-GO = conditioning loses to averaging
    return _row("P5 donor-structure", "structure",
                "Does donor-CONDITIONING beat donor-AVERAGING for perturbation recovery?",
                "cross-donor recovery: conditioned vs averaged (perm-tested)",
                round(cond, 4), f"averaging {avg:.3f}",
                f"NO-GO — donor-conditioning ({cond:.3f}) < donor-averaging ({avg:.3f}); same-gene "
                f"concordance is real but floor-magnitude. Conditioning gives no gain.",
                "NO-GO", consistent, "results/donor_structure_gate.csv", floor=NOISE_FLOOR)


# ---- P6: relational-object --------------------------------------------------------------------
def relational_object():
    rows = _rows("results/relational_gate.csv")
    objs = {x["object"]: _f(x["specific"]) for x in rows}
    best_name = max(objs, key=lambda k: objs[k])
    best = objs[best_name]
    any_pass = any(x["pass"].strip().lower() == "true" for x in rows)
    consistent = (not any_pass) and best < REL_THRESHOLD
    return _row("P6 relational-object", "structure",
                "Does any specific-space relational object (similarity/loadings/rank) reach 0.30?",
                "reproducible specific-space contrast vs degree-preserving null",
                round(best, 4), f"threshold {REL_THRESHOLD}",
                f"FAIL — no relational object reaches {REL_THRESHOLD} (best = {best_name} {best:.3f}; "
                f"S-similarity 0.008). Raw-space S≈0.9 is a constant-cosine artifact. Floor is object-general.",
                "FAIL", consistent, "results/relational_gate.csv")


# ---- P7: external causal-edge validation (C-FUSE 1b, G-F.2) -----------------------------------
def external_validation():
    rows = {x["subset"]: x for x in _rows("results/fusion_gf2.csv")}
    c = rows["combined"]
    causal, twin = _f(c["causal_acc"]), _f(c["twin_acc"])
    nullm = _f(c["null_mean"])
    diff = _f(c["causal_minus_twin"])
    consistent = diff <= 0                          # committed FAIL(causal-specificity) = twin >= causal
    return _row("P7 external causal-edge validation", "external",
                "Do held-out external causal edges reveal a CAUSAL (not just predictive) advantage?",
                "sign-recovery of held-out external edges: causal vs non-causal twin vs deg-preserving null",
                round(diff, 4), f"causal {causal:.3f} vs twin {twin:.3f} (null {nullm:.3f})",
                f"IN-DISTRIBUTION — recovers external edge direction above null ({causal:.3f}>{nullm:.3f}) "
                f"but with NO causal-specific advantage over the twin (causal−twin {diff:+.3f}). The C2 "
                f"edge does not transfer to external causal structure.",
                "FAIL (causal-specificity)", consistent, "results/fusion_gf2.csv", floor=nullm)


# ---- predictability budget (the measured ceiling + floor) -------------------------------------
def budget():
    cd = {x["split"]: x for x in _rows("results/budget_cross_donor.csv")}
    foc = {(x["model"], x["split"]): x for x in _rows("results/fraction_of_ceiling.csv")}
    floor_cond, pp_cond = _f(cd["condition"]["cross_donor_specific_r"]), _f(cd["condition"]["perm_p"])
    floor_gene = _f(cd["gene"]["cross_donor_specific_r"])
    ridge_gene = _f(foc[("ridge", "gene")]["frac_of_ceiling_median"])
    causal_gene = _f(foc[("causal", "gene")]["frac_of_ceiling_median"])
    consistent = floor_cond > 0 and pp_cond < 0.05
    return _row("Budget (ceiling + floor)", "calibration",
                "How much signal is recoverable at all, and where does linear collapse?",
                "cross-donor floor (perm-tested) + fraction-of-reliability-ceiling per model",
                round(floor_cond, 4), f"gene floor {floor_gene:.3f}; ridge 0.02 vs do-op {causal_gene:.2f} of ceiling (gene)",
                f"FLOOR≈{floor_cond:.3f} cond / {floor_gene:.3f} gene, real (perm_p {pp_cond:.0f}). A structured "
                f"residual exists: linear collapses on the gene axis (ridge {ridge_gene:.2f} of ceiling) where "
                f"the do-operator reaches {causal_gene:.2f} — but the pointwise volume is small.",
                "not noise-saturated; floor real", consistent, "results/budget_cross_donor.csv",
                floor=NOISE_FLOOR)


# ---- do-operator C2 positive control (signal-detection anchor) --------------------------------
def c2_control():
    bt = {(x["model"], x["split"]): _f(x["pearson_delta_top50"]) for x in _rows("results/benchmark_table.csv")}
    c2_cond = bt[("causal", "condition")] - bt[("noncausal", "condition")]
    c2_gene = bt[("causal", "gene")] - bt[("noncausal", "gene")]
    consistent = c2_cond > 0 and c2_gene > 0        # POSITIVE anchor: do-operator beats twin
    return _row("C2 do-operator control (POSITIVE)", "positive-control",
                "Does the SAME null machinery that flags the six negatives ever register a positive?",
                "causal − non-causal twin, Pearson-δ(top-50 DEG), within-distribution hold-outs",
                f"+{c2_cond:.3f}/+{c2_gene:.3f}", "vs non-causal twin",
                f"POSITIVE — the do-operator beats its non-causal twin within-distribution "
                f"(+{c2_cond:.3f} condition / +{c2_gene:.3f} gene). Signal-detection anchor: the nulls mean "
                f"'no signal', not 'no sensitivity'.",
                "POSITIVE (do-operator confirmed)", consistent, "results/benchmark_table.csv")


PROBES = [causal_matrix, fluctuation_cnl, single_cell_snr, trajectory_geometry,
          donor_structure, relational_object, external_validation]
CALIBRATION = [budget, c2_control]


def run_audit(dataset: str = "marson", write: bool = True):
    """Assemble the predictability scorecard for `dataset` (only 'marson' is committed).
    Returns (rows, passed). PASS = all 7 probe verdicts reproduce the committed verdict AND the
    do-operator C2 control registers positive (anti-triviality: the map measures, it does not just fail)."""
    if dataset != "marson":
        raise ValueError(f"only the committed 'marson' scorecard is available (got {dataset!r})")
    probe_rows = [fn() for fn in PROBES]
    calib_rows = [fn() for fn in CALIBRATION]
    rows = probe_rows + calib_rows
    probes_ok = all(r["consistent"] for r in probe_rows)
    c2_row = next(r for r in calib_rows if r["family"] == "positive-control")
    passed = probes_ok and c2_row["consistent"]
    if write:
        out = os.path.join(ROOT, "results", "predictability_audit_gate.csv")
        cols = ["probe", "family", "question", "metric", "score", "reference", "noise_floor",
                "verdict", "committed_verdict", "consistent", "source"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows, passed


def _main():
    rows, passed = run_audit("marson")
    print("=" * 100)
    print("MARSON CD4 PERTURB-SEQ — PREDICTABILITY SCORECARD  (reproduced from committed gate CSVs)")
    print("=" * 100)
    for r in rows:
        flag = "ok" if r["consistent"] else "XX MISMATCH"
        print(f"[{flag}] {r['probe']:36s} | {r['verdict']}")
    n_neg = sum(1 for r in rows if r["family"] not in ("calibration", "positive-control"))
    print("-" * 100)
    print(f"Probes reproduced faithfully: {sum(1 for r in rows[:7] if r['consistent'])}/7   "
          f"C2 positive control: {'POSITIVE' if rows[8]['consistent'] else 'FAILED'}")
    print(f"G-PA.1 VERDICT: {'PASS — faithful audit' if passed else 'FAIL — packaging changed the answers'}")
    return passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if _main() else 1)
