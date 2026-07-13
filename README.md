# CD4+ T Cell Perturbation Prediction — Causal + JEPA

**Built with Claude: Life Sciences (2026).**

Predicting the transcriptional effect of a CRISPRi gene knockdown in primary human
CD4+ T cells — including in an **activation state the model has never seen**.

> "Standard models treat a gene knockdown as an *observation*. We treat it as an
> *intervention* — and that distinction is what lets the model predict a
> knockdown's effect in an activation state it has never seen."

This repository is one pipeline whose experimental core is a **2×2 ablation**
(`JEPA-init × causal-mask`). The causal claim, the do-operator isolation, and the
JEPA claim are three cells of that single matrix. Ridge / TabPFN / PseudoBulk-FCN /
Arc State are external reference points; a Value-of-Information (VOI) layer turns
model disagreement into a sample-efficient experimental-design recommendation.

The full technical specification is [`UNIFIED_BUILD_PLAN.md`](docs/UNIFIED_BUILD_PLAN.md).
The pre-registered hypotheses are in [`hypotheses.md`](hypotheses.md) and were
committed **before** any model saw data.

> ### Submission of record — v2: The Predictability Audit
> The promoted submission reframes this work as a **dataset predictability scorecard** — the seven
> pre-registered probes + the predictability budget + the do-operator positive control, each scored against a
> degree/label-preserving null and calibrated to the measured reliability ceiling. It is an
> **evaluation/methods contribution** (novelty **Tier-2**), **not a new predictor** — and it is honest to its
> bounds: the do-operator's one accuracy positive (C2) is **in-distribution, not causal** (it does not
> replicate on held-out external causal edges), and a second-dataset port shows the *machinery* ports, **not**
> that the floor finding generalizes. Full story: [`PREDICTABILITY_AUDIT.md`](PREDICTABILITY_AUDIT.md).
>
> **Releases:** `submission-v2` (@ `a8878d5`) is the submission of record; `submission-fallback-v1`
> (@ `6476670`) is the frozen, reversible fallback.

## Results at a glance

- **Headline (do-operator, C2).** The interventional causal mask beats its non-causal twin on
  zero-shot perturbation prediction — Pearson-δ (top-50 DEGs) **+0.118** on the condition hold-out
  and **+0.162** on the gene hold-out ([`results/benchmark_table.csv`](results/benchmark_table.csv):
  causal 0.344/0.368 vs non-causal 0.226/0.206). On a **fraction-of-ceiling** axis (the honest metric —
  raw δ is baseline-dominated) the do-operator reaches **0.55** of the achievable gene-axis signal,
  where a linear model collapses to **0.02** ([`BUDGET.md`](docs/BUDGET.md)).
- **The frontier is mapped, honestly.** Six *pre-registered* CPU gates asked whether the
  per-perturbation prediction floor (~**0.03** cross-donor) can be broken — causal-matrix, fluctuation,
  single-cell SNR, trajectory-geometry, donor-structure, relational structure. **All six are clean
  negatives**, each traceable to a committed gate CSV, **zero GPU spent**. The floor is real and
  object-general. A **seventh, GPU-gated** external-validation test (C-FUSE 1b) then closed the causal
  question — the do-operator's within-dataset C2 advantage does **not** replicate on held-out external
  causal edges (recovered above null, but no better than its non-causal twin), so the edge is
  **in-distribution, not causal** (data-integrity C2 control passed). → [**Supplementary analyses — the full arc**](#supplementary-analyses--the-full-arc).

---

## Dataset

Marson/Pritchard genome-scale CRISPRi Perturb-seq in primary human CD4+ T cells
(~22M cells, every expressed gene silenced one at a time, 4 donors, 3 activation
states: Rest / Stim8hr / Stim48hr).

- **GEO:** GSE278572 · CZI Virtual Cells mirror · bioRxiv `10.64898/2025.12.23.696273`
- **CP1 data source:** `GWCD4i.pseudobulk_merged.h5ad` (44.6 GB pre-computed pseudobulk;
  278,684 guide×donor×condition profiles × 18,129 genes). CP1 runs on pseudobulk deltas;
  the ~1.7 TB of single cells are the JEPA lane's input.
- **Split SHA256 (frozen):** `fd2b8c21d357f8699ec34e2d5ebc1639612c27a0147a9ca94d4983822d93247e`
  — binds the split to that exact file; every module verifies it at startup.

## The claims (pre-registered — see [`hypotheses.md`](hypotheses.md))

| ID | Claim |
|----|-------|
| **C1** | The `CausalCisTransFormer` (corrected do-mask) matches or beats strong baselines **including TabPFN** on the **condition hold-out** (zero-shot Stim48hr). |
| **C2** | The causal mask beats its non-causal twin on the condition hold-out (do-operator isolation). Reported regardless of leaderboard position. |
| **C3** | JEPA-init helps the condition hold-out (full 2×2, JEPA × causal). |
| **S1** | Model-disagreement VOI ranks which perturbations are most worth measuring; VOI-guided selection reaches ~90% of full-screen accuracy from a fraction of perturbations. |

**The 2×2 (experimental core):**

| Encoder init | Causal mask | Label |
|---|---|---|
| Random | off | Direct-regression baseline (`noncausal`) |
| Random | on  | Causal-only — C2 treatment (`causal`) |
| JEPA   | off | JEPA-only (`jepa_only`) |
| JEPA   | on  | **JEPA + causal — main model** (`jepa_causal`) |

## CP1 results (real data, L4)

Trained on the frozen split (SHA above): 2,269 / 318 evaluable HVG-panel perturbations on the
condition / gene hold-outs. Headline metrics (`results/benchmark_table.csv`; full 8-metric
appendix in `results/benchmark_table_full.csv`):

| split | model | Pearson-δ (top-50) ↑ | PerturBench rank ↓ | DES ↑ |
|---|---|---|---|---|
| **condition** | **causal** | **0.344** | 0.457 | 0.587 |
| condition | non-causal | 0.226 | 0.483 | 0.579 |
| condition | ridge | 0.384 | **0.365** | 0.651 |
| condition | fcn | 0.086 | 0.500 | 0.535 |
| **gene** | **causal** | **0.368** | 0.440 | 0.599 |
| gene | non-causal | 0.206 | 0.484 | 0.590 |
| gene | ridge | 0.019 | 0.501 | 0.506 |
| gene | fcn | 0.107 | 0.500 | 0.554 |

**C2 — the do-operator works (headline, pre-registered).** Same architecture, mask on vs off:
the causal mask beats its non-causal twin by **+52%** on the condition hold-out (0.344 vs 0.226)
and **+79%** on the gene hold-out (0.368 vs 0.206). The advantage is consistent across
Pearson-δ, E-distance (2.46 vs 6.46 condition; lower better), and AUPRC. The corrected
do-mask — masking only the perturbed gene's query row so the intervention propagates
downstream — is doing real work.

**Zero-shot to unseen genes:** the causal model generalizes to genes it never saw silenced
(**0.368**) where the linear baseline fully collapses (Ridge **0.019**).

**Honest caveats.** On the pure condition shift, a simple gene→δ linear map (Ridge, 0.384) is
still competitive with / slightly ahead of the causal transformer (0.344), and only Ridge on the
condition hold-out clears the mode-collapse bar (rank < 0.4) — the transformers sit in the
borderline 0.44–0.48 band (causal always sharper than non-causal). TabPFN is license-gated on a
headless box (see RUNBOOK); the JEPA cells of the 2×2 (C3) are CP2.

## The two corrections this build refuses to revert (`UNIFIED_BUILD_PLAN.md` §1)

1. **Causal do-mask propagates.** An intervention removes only edges *into* the
   perturbed gene (mask its query row); other genes **must still attend to it** so
   the intervention propagates downstream. We do **not** add `M[:, perturbed] = -inf`.
   (DoFormer, bioRxiv 2026.05.02.722054.)
2. **JEPA uses an EMA teacher at single-cell resolution.** Student on masked input
   + stop-gradient EMA teacher on unmasked input + predictor head + cosine loss,
   masking expression *values* within a cell — not a pseudobulk MLP.
   (Cell-JEPA, arXiv 2602.02093.)

## Repository layout

```
cd4-perturb-causal-jepa/
  README.md                    # this file
  PREDICTABILITY_AUDIT.md      # v2 submission of record — the predictability scorecard
  GPA2_PORT.md                 # second-dataset port (Schmidt 2022) — machinery ports, qualified
  RESULTS.md                   # detailed CP1/CP2 results
  SUBMISSION_FALLBACK.md       # v1 frozen-fallback summary
  hypotheses.md                # pre-registration (committed before Day 1)
  predictability_audit/        # run_audit() — the scorecard package (stdlib-only, no retrain)
  core/                        # frozen contract, data, pseudobulk, features, split, eval, models/, voi
  scripts/                     # gate + build scripts (run_cp1, fusion_*, gpa2_*, make_scorecard_figure, …)
  mechanism/                   # Â_C + fluctuation (C-NL) probes + findings
  results/                     # committed gate CSVs (the source of truth)
  figures/                     # committed figures (predictability_scorecard.svg, …)
  explorer/                    # offline demo bundle
  docs/                        # the full arc — BUDGET, PHASEB, DONOR, RELATIONAL, TRAJECTORY,
                               #   FUSION_GATES, CP2_SUMMARY, RUNBOOK, HANDOFF, UNIFIED_BUILD_PLAN, DEV2_NOTES
  tests/                       # unit + regression tests
  split_manifest.json · split/ # immutable split + frozen HVG list
  Snakefile · environment.yml · requirements-dev.txt · gpu_queue.py · run_fusion_pipeline.sh
```

**Shared artifacts live outside git** in `DATA_ROOT` (default `~/cd4-perturb-data/`,
override with `CD4_DATA_ROOT`): `embeddings/`, `pseudobulk/`, `features/`, `cells/`,
`runs/`, `checkpoints/`, and the `GPU_LOCK`. Only code, `split_manifest.json`,
`hypotheses.md`, `results/benchmark_table.csv`, and `figures/` are committed. See
`core/contract.py` for every canonical path.

## Reproducing

```bash
# 1. environment
conda env create -f environment.yml && conda activate cd4-perturb

# 2. run the whole pipeline (Lane C on CPU; GPU jobs go through the queue)
snakemake --cores all

# GPU training/inference is never launched directly — always via the serial queue:
python gpu_queue.py submit esm2      # G1
python gpu_queue.py submit causal    # G2
python gpu_queue.py submit noncausal # G3
# ... in the §6 priority order; each job runs the epoch-1 measure-then-extrapolate gate first.
```

## Single-GPU concurrency (`UNIFIED_BUILD_PLAN.md` §6)

Three lanes, one L4 (24 GB). **Lane C** (CPU) runs data/QC/pseudobulk/features/eval
continuously and scores each model the moment its `runs/*.parquet` lands. **Lane G**
is a *serial* GPU queue (`gpu_queue.py`) — one job at a time, in priority order, each
gated by a 1-epoch measure-then-extrapolate check. **Lane D** is code-writing
concurrency across up to two `git worktree` checkouts that share the one queue.

## Acceptance checkpoints

- **CP1** — `results/benchmark_table.csv` has `ridge`, `tabpfn`, `fcn`, `causal`,
  `noncausal` on gene + condition hold-out; split SHA verified; eval harness passing.
- **CP2** — 2×2 complete (`jepa_only`, `jepa_causal`); VOI + subsampling curve; three
  demo figures; `snakemake --cores all` end-to-end.
- **CP3** — reproducibility package tagged; full 8-metric appendix; one-command rerun.

**Split integrity.** The frozen split is SHA-bound: `core.split.verify()` checks `split_manifest.json`
against SHA `fd2b8c21…`. `snakemake -n cp2` resolves the DAG end-to-end.

**Known non-blocking test flake.** `test_ridge_learns_and_records` asserts a synthetic
`ridge Pearson-δ > 0.4` bound that numpy 2.4.6 (in-spec per the `numpy=2.*` pin) trips; the suite is
otherwise **69/70 green** and **CP2's committed numbers are unaffected** (`results/benchmark_table.csv`
is byte-identical since the CP2-final commit). Tracked as a follow-up (loosen the synthetic threshold or
tighten the pin) — not a regression in any pipeline code.

## Supplementary analyses — the full arc

Beyond the primary CP1/CP2 benchmark, a sequence of pre-registered supplementary analyses probe *why*
the hard perturbations are hard. Each is committed with an honest one-line verdict and a pointer;
every claim is **fraction-of-ceiling / partial-correlation, per axis** (raw Pearson-δ is
baseline-dominated), and results are framed as attempts against a measured target, not solved problems.

**Provenance trail (auditable):** every result carries a per-result git **tag** (`cp1`, `cp2`,
`budget-final`, `phaseB-final`, `mechanism-spike-final`, `cnl-gate-final`, `cnl-realdata-final`,
`trajectory-final`, `donor-final`, `relational-final`, `core-frozen`, `fusion-gf2-final`), a
**pre-registration** in [`hypotheses.md`](hypotheses.md) (committed before the data was seen, with the
go/no-go thresholds), and a committed **gate CSV** under `results/` (or `mechanism/results/`). The seven
negatives are reproducible look-ups, not recollections.

| Analysis | One-line verdict | Pointer |
|---|---|---|
| **Primary — CP1/CP2** | do-operator (C2) confirmed: causal beats its non-causal twin **+0.118 condition / +0.162 gene**; JEPA 2×2 + VOI | [`RESULTS.md`](RESULTS.md), [`CP2_SUMMARY.md`](docs/CP2_SUMMARY.md), `results/benchmark_table.csv` |
| **Predictability budget** | benchmark is **not noise-saturated**; bucket C (structured, gene ≈0.76 of ceiling) is real (cross-donor perm p<0.001); report fraction-of-ceiling per axis | [`BUDGET.md`](docs/BUDGET.md) |
| **Residual localization + SNR pre-check** | the unrecovered residual **is the transient activation-cytokine program** (IFNG/IL2/CSF2/chemokines, peaks Stim8hr); the per-perturbation frontier is a **confirmed noise floor** at pseudobulk depth — single-cell resolution would not fix it (no 130 GB spent) | [`PHASEB.md`](docs/PHASEB.md) |
| **Mechanism line** | Â_C (spikes #1/#2) **FAIL** under P≪G; C-NL third-moment gate **positive** on the simulator; C-NL real-data **NEGATIVE** (third moment orthogonal, 12/12 strata) | [`mechanism/`](mechanism/), [`mechanism/FINDINGS_CNL_REALDATA.md`](mechanism/FINDINGS_CNL_REALDATA.md) |
| **Trajectory-coupling** | **clean negative** — recoverability is *not* explained by trajectory-geometry (partial ρ ≈ 0, both splits), and the reduced scalar target sits at the noise floor; no build ran | [`TRAJECTORY.md`](docs/TRAJECTORY.md) |
| **Donor-structured recovery** | **NO-GO** — within-donor same-gene concordance is real but at noise-floor magnitude (Δ≈0.017); donor-*averaging* beats donor-*conditioning* (0.034 vs 0.016); the floor is real, reversal refuted on the dataset's own 2-guide design | [`DONOR.md`](docs/DONOR.md) |
| **Relational-object recovery** | **FAIL** — relational structure is floored too: no specific-space object (similarity/loadings/rank) reaches 0.30 (S 0.008, best loading factor 0.17, high-effect subset 0.037). Raw-space S ≈ 0.9 is a *constant-cosine artifact*, not a reproducible pattern (repo measures 0.007). The floor is object-general | [`RELATIONAL.md`](docs/RELATIONAL.md) |
| **External causal-edge validation (C-FUSE 1b, GPU)** | **FAIL (causal-specificity)** — the do-operator recovers held-out external edge *direction* above null (9/9 regulators >0.5, binom p=0.004) but with **no** advantage over its non-causal twin (causal−twin **−0.010**, regulator cluster-bootstrap CI [−0.013, −0.005]); the within-dataset C2 edge is **in-distribution, not causal**. Signal rides Freimer *indirect* KO-DE edges; the 45 *direct* Weinstock (LLCB) edges sit at chance (0.400, p=0.94). Data-integrity **C2 positive control passed** (restored data verified real: +0.106/+0.156 vs committed +0.118/+0.162) | [`FUSION_GATES.md`](docs/FUSION_GATES.md) |

**Provenance notes (preserved across the arc):** the CIPHER raw-count residual ≠ the budget's Ridge-based
bucket C — a looser object; only its *structure* transfers. The third-moment link is an *inference* from
response theory, not a CIPHER claim. Reliability is an *upper bound* on any model's achievable δ, not a
guarantee a model reaching it exists.

### Mechanism recovery

A reproducible synthetic study ([`mechanism/`](mechanism/), CPU-only) testing whether explicit
per-context causal-matrix (`Â_C`) estimation beats correlation baselines for cross-context
transportability. **Result: it does not, under `P≪G`** — in either the linear/single-perturbation or
the nonlinear/double-perturbation regime (pre-registered bar not met → FAIL, honestly reported). It
documents *why* correlation baselines are so hard to beat on the field's own simulator (the stationary
covariance solves the Lyapunov equation, so `Σ` is a near-sufficient statistic for `A`), and — a
standalone positive — how the linear transportability condition itself degrades (AUROC 1.00 → 0.88) as
the system becomes nonlinear. The transportability signal is real (oracle with true `A` = 1.0) but
estimation-gated; the one un-tested lever is a materially better estimator.

A **third probe (the C-NL gate) is the positive** of the line: covariance/Lyapunov sufficiency is a
*second-moment* property, so the one signal it provably cannot carry lives in the third moment. On
ground truth the baseline **third moment predicts the second-order perturbation response covariance
misses** — ΔR² ≈ +0.6–0.75 (CI excluding 0), surviving to 1,000 control cells with NB emission on. The
term is small (~3–4% of the response) but strongly structured; sizing it on real CD4 data is the
go/no-go. Provenance-guarded (the third-moment link is an inference from response theory, not a CIPHER
claim). See [`mechanism/README.md`](mechanism/README.md) and [`mechanism/FINDINGS_CNL.md`](mechanism/FINDINGS_CNL.md).

That go/no-go has now been run, and it **closes the line as a negative**: on real CD4⁺ CRISPRi data
(4 donors × 3 states, 16,188 perturbations, CIPHER-exact raw counts) the baseline third moment is
**orthogonal** to the first-order residual — ΔR² pinned at zero in all 12 strata (mean +0.0000, jackknife
95% CI [−0.0000, +0.0000]), feature well-formed everywhere. Room exists (the linear model leaves ~91% of
variance unexplained, and its fit degrades further under stimulation) but the third moment fills none of
it. The simulator signal does not survive real single-cell estimation error → **no-go on a third-moment
closed-form.** Full readout: [`mechanism/FINDINGS_CNL_REALDATA.md`](mechanism/FINDINGS_CNL_REALDATA.md).

### Predictability budget

A CPU-only partition of each perturbation's response into **B** irreducible noise (0.22 gene / 0.33
condition), **A** linear-explainable, and **C** structured residual, on the eval metric's own scale. The
ceiling is far from saturated; bucket C is large on the gene axis (≈0.76) and **real** — the
perturbation-specific residual reproduces across donors ~60–100× above a shuffled-label null (perm
p<0.001). The do-operator recovers ~56% of it; ~44% is a located gap. Reframes the benchmark as
**fraction-of-ceiling, per axis**. Full readout: [`BUDGET.md`](docs/BUDGET.md).

### Residual localization + single-cell SNR pre-check

The bucket-C residual is localized to the **transient T-cell activation cytokine program** (IFNG, IL2,
CSF2, IL3/IL13, CCL/CXCL chemokines, IL2RA) that peaks at the Rest→Stim8hr transition (4/4 donors,
effect/confound-controlled) — a far-from-equilibrium response. Its reproducible part is a *shared*
program; the *per-perturbation* residual is noise-limited at pseudobulk (cross-donor ~0.03). A
pre-registered SNR pre-check asked whether single-cell resolution would recover it and found **no** —
single-cell adds no cells (pseudobulk is sufficient for the mean); reaching a usable floor needs ~12×
cells or ~8% concentration, projected best-case ~0.10. A **confirmed noise floor**, reached with zero
GPU/egress. Full readout: [`PHASEB.md`](docs/PHASEB.md).

### Trajectory-coupling (clean negative)

Tested whether predictability is a *dynamical-geometry* property — unrecoverable perturbations moving the
cell *along* the activation axis, reducible to a scalar displacement. A two-part CPU gate (pre-registered
C-TC.1/C-TC.2 in [`hypotheses.md`](hypotheses.md)) **both fail**: recoverability vs trajectory-coupling
partial Spearman(R,TC | magnitude, reliability) = **+0.007 (condition), +0.034 (gene)**, p = 0.75/0.55
(bar was |ρ|≥0.3) — 1D and 2D; and the scalar target reproduces at ~0.07, at the noise floor (random-axis
null 0.03). Predictability is **not** a trajectory-geometry property here; no build ran. The gate projects
onto a *measured* axis only — it fits **no** dynamical model (3 timepoints underdetermine a vector field).
Full readout: [`TRAJECTORY.md`](docs/TRAJECTORY.md).

### Donor-structured recovery (fifth clean negative)

Tested whether the per-perturbation floor is a cross-donor-*averaging* artifact — a memo showed 0.48
within-donor vs 0.03 cross-donor reproducibility (16× gap), suggesting recoverable donor-specific
structure. A two-part CPU gate (pre-registered C-DON.1/C-DON.2 in [`hypotheses.md`](hypotheses.md)),
run against the dataset's own **2-gRNA/gene** design, **both fail.** Step 0 confirmed the two guides
per gene are batch-orthogonal (span both 10x runs), so the test is valid. **G-D.1:** same-gene
independent-guide within-donor concordance *is* significant (Δ≈0.017, 4/4 donors, perm p<0.001,
survives composition-correction) — real target-specific biology, but ~8× below the Δ≥0.15 bar, i.e. at
noise-floor magnitude. **G-D.2:** donor-conditioned recovery (0.016) is beaten by donor-*averaging*
(0.034) and even wrong-donor (0.024) — donor-conditioning gives *negative* gain. The "0.48" was a
noise-*model* estimate, not empirical; the 16× gap **inverts** (averaging helps). The floor is real;
the reversal is refuted. No build (G13 unlicensed). Full readout: [`DONOR.md`](docs/DONOR.md).

### Relational-object recovery (sixth clean negative)

The five negatives above all scored the *pointwise* per-perturbation delta δ_p (cross-donor
reproducibility ~0.03). This tested a **different object** — the *relational* structure over
perturbations (perturbation×perturbation similarity **S**, program loadings **L**, per-gene rank
**R**), which averages over many genes so per-cell noise averages out. Run in **specific**
(shared-program-removed) space, because raw similarity is dominated by the shared activation program
(pre-registered C-REL.1/C-REL.2 in [`hypotheses.md`](hypotheses.md)). **G-R.1 fails:** no specific-space
object reaches the 0.30 bar — S = **0.008**, L top-3 = **0.11** (best factor 0.17, above its 0.019 null
but ≪ 0.30), R = **0.025**; restricting to the top-200 high-effect perturbations lifts S only to
**0.037**, so the floor is *total*, not population-dilution. The machinery is calibrated (it reproduces
the committed 0.049 pointwise floor and detects the loading whisker). **Honest correction:** raw-space
S reproducibility is **0.007**, not the ~0.9 a shared-program tautology would suggest — that ~0.9 is
the constant high *baseline* of raw cosines (all perturbations point toward the shared program), not a
reproducible *pattern*. G-R.2 (known-regulator recovery) was gated on a pass and not run; the build
(G14) is unlicensed. The frontier's noise floor is **object-general** — pointwise *and* relational, raw
*and* specific, whole-population *and* high-effect. Full readout: [`RELATIONAL.md`](docs/RELATIONAL.md).

## License

[MIT](LICENSE).
