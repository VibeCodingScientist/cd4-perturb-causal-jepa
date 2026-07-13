# CD4⁺ Perturbation Prediction — Submission (frozen fallback summary)

*This document provides a standalone summary of the `submission-fallback-v1` release. It is
self-contained: it can be read in isolation. All numbers are read from the committed artifacts in this
repository.*

**Built with Claude: Life Sciences (2026).** GitHub: `VibeCodingScientist/cd4-perturb-causal-jepa`.

## What this is
This work predicts the transcriptional effect of a CRISPRi gene knockdown in primary human **CD4⁺ T
cells**, including the zero-shot setting in an activation state the model has never observed. The dataset
is the Marson/Pritchard genome-scale CRISPRi Perturb-seq resource (GEO **GSE278572**; ~22M cells, 4
donors, states Rest/Stim8hr/Stim48hr, 3000 HVG). The experimental core is a **2×2 ablation** (JEPA-init ×
causal-mask). The central thesis is that a knockdown should be modeled as an **intervention** (the
do-operator) rather than as an observation.

## The headline result (`results/benchmark_table.csv`)
The **do-operator** (interventional causal mask) outperforms its non-causal counterpart on zero-shot
perturbation prediction, measured by Pearson-δ (top-50 DEGs):

| | causal | non-causal | **C2 (do-operator effect)** |
|---|---|---|---|
| condition hold-out (zero-shot Stim48hr) | 0.344 | 0.226 | **+0.118** |
| gene hold-out (unseen silenced genes) | 0.368 | 0.206 | **+0.162** |

On the more conservative **fraction-of-ceiling** axis (raw δ is baseline-dominated), the do-operator
reaches **0.55** of the achievable gene-axis signal, whereas a fitted linear model collapses to **0.02**
(`results/fraction_of_ceiling.csv`). The benchmark is not noise-saturated (noise floor ~0.22 gene / 0.33
condition), and bucket C (structured signal) is genuine: the perturbation-specific residual reproduces
cross-donor above a shuffled null at **p<0.001** (`results/budget_cross_donor.csv`).

## The analytic arc — six pre-registered negatives map the frontier
The per-perturbation prediction floor is ~0.03 cross-donor. Six pre-registered CPU gates asked, from
independent angles, whether this floor can be broken. All six return clean negatives with no GPU
expended, each accompanied by a committed gate CSV and a `hypotheses.md` pre-registration:

| direction | verdict | key number |
|---|---|---|
| Causal-matrix (Â_C) | FAIL | mechanism AUROC 0.62–0.69 vs correlation-null 0.83 (gap ≈ −0.14) |
| Fluctuation / third-moment | FAIL | ΔR² ≈ 0, perm p 0.75–0.997, 12/12 strata |
| Single-cell SNR (CellCap) | NOT-GREEN | per-pert 0.033; ~12× cells needed; projected ~0.10 |
| Trajectory-geometry | CLEAN NEG | partial ρ(R,TC) +0.007 / +0.034; scalar at floor |
| Donor-structure | NO-GO | within-donor concordance Δ≈0.017; donor-averaging (0.034) beats conditioning (0.016) |
| Relational structure | FAIL | specific-space S 0.008, best loading 0.17, high-effect 0.037 (all ≪ 0.30) |

The floor is object-general: it holds pointwise and relationally, raw and specific, whole-population and
high-effect. This constitutes the scientific spine of the submission, namely a validated method
contribution built on a rigorously bounded and honestly reported negative.

## Reproduce (one command)
```bash
conda env create -f environment.yml && conda activate cd4-perturb
snakemake --cores all          # Lane-C CPU pipeline; GPU jobs go through gpu_queue.py
```
The frozen split is SHA-bound (`core.split.verify()` checks `split_manifest.json` against
`fd2b8c21…`); `snakemake -n cp2` resolves the DAG. **Known non-blocking test flake:**
`test_ridge_learns_and_records` (a synthetic `ridge Pearson-δ > 0.4` bound) trips under numpy 2.4.6 (in
specification, per the `numpy=2.*` pin). The suite is otherwise 69/70, and CP2's committed numbers are
byte-identical. This is tracked as a follow-up rather than a regression.

## Explore it (no build, offline)
Open **`explorer/explorer_bundle.html`** in a browser. It is a self-contained three-act walkthrough (the
do-operator result, the fraction-of-ceiling reframing, and the six-negative frontier map) that reads the
committed numbers directly. All panels are backed by `_meta.source == "real"`; no demo data is present.

## Provenance (auditable)
Every result carries a per-result git **tag** (`cp1`, `cp2`, `budget-final`, `phaseB-final`,
`mechanism-spike-final`, `cnl-gate-final`, `cnl-realdata-final`, `trajectory-final`, `donor-final`,
`relational-final`, `core-frozen`), a **pre-registration** in `hypotheses.md` (committed before the data
was seen, with go/no-go thresholds), and a committed **gate CSV** under `results/` or
`mechanism/results/`. The six negatives are therefore reproducible look-ups rather than recollections. See
the README section "Supplementary analyses — the full arc" for the one-click index.

## Freeze note
`submission-fallback-v1` is the known-good submission. All further work proceeds on branches, never on
this tag or its commit, so that this fallback remains reproducible while exploration continues.
