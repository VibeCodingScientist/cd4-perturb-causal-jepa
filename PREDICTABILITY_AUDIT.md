# The Predictability Audit — a scorecard reframe of the Marson CD4 Perturb-seq work (v2 — submission of record)

*Developer 2. This is the **v2 submission of record**: an honest reframe of everything already validated —
the seven pre-registered probes + the predictability budget + the do-operator positive control — assembled
into a **dataset predictability scorecard**. It is **not an eighth model** (seven investigations proved that
model does not exist at this depth); it is an **evaluation/methods reframe** of existing, committed,
validated content. **v2 was promoted** (PR #12 merged to `main`; release `submission-v2` @ `a8878d5` cut as
the submission of record); the frozen **v1** (`submission-fallback-v1` / `6476670`) is kept byte-untouched as
the reversible fallback.*

---

## What this is (and is not)
- **Is:** a reproducible **audit** that measures *how predictable the Marson CD4 dataset actually is* — the
  intrinsic ceiling, the noise floor, and which kinds of structure (pointwise, mechanistic, donor,
  relational, trajectory, external-causal) carry recoverable signal vs. which sit at the floor — each scored
  against a degree/label-preserving null, with the do-operator as a signal-detection anchor.
- **Is not:** a new predictor. If the hackathon strictly rewards a more accurate model, **this is not that**
  — that model was proven not to exist here, seven times. It wins on the axis the field has *explicitly
  asked for* (assessing dataset/causal validity beyond accuracy) and on the honesty of the negative-space
  map. **Whether that axis counts is the lead's call.**

## Paradigm lineage (cited, not invented)
Intrinsic **predictability / forecastability** is an established paradigm — the property of a *system/dataset*
that bounds any model, distinct from any one model's error. It comes from ecology and time-series
forecasting (e.g. Pennekamp et al., *the intrinsic predictability of ecological time series*; spectral-entropy
forecastability measures). **We claim only the instantiation** — porting that paradigm to a Perturb-seq
dataset — **not the concept.** *(Exact paradigm citations to be firmed up before any external use; provided
as a pointer by the brief.)* This is **not** "ImageNet/GLUE for cells": it is a validated audit on one
(possibly two) dataset(s) — a seed, not a finished standard.

## Step 0 — occupancy check → **novelty Tier-2** (conservative)
Full-text/preprint reads of the four most-likely occupiers (parallel agents; sources below). The question:
does any of them profile a **dataset's own intrinsic predictability** (a ceiling/audit), vs. ranking models?

| Work | What it does | Profiles dataset predictability? |
|---|---|---|
| **scPerturBench** (Nat. Methods 2025, 10.1038/s41592-025-02980-0) | Ranks **27 models** across 29 datasets on 6 model-vs-truth metrics | **No** — model ranking; no per-dataset ceiling (full text paywalled, so not 100% excluded) |
| **"A Systematic Comparison…"** (bioRxiv 10.1101/2024.12.23.630036) | Separate model benchmark (12 methods, 25 datasets) | **No** — attributes failure to "heterogeneity" but computes no ceiling |
| **CIPHER** (Kuznets-Speck et al., bioRxiv 2025.06.27.661814) | Linear-response model; predicts outcomes from baseline covariance | **Ambiguous** — correlates a per-dataset structural property (participation ratio) with predictability (R²=0.75, slope −6.5) across 11 datasets |
| **"Virtual Cells as Causal World Models"** (OpenReview qjIq4JWFVs) | Proposes a causal-eval taxonomy; **calls for** benchmarks | **No — and cited as the field's own request** for exactly this |
| **PerturbPlan** | Pre-data experiment **design/power** tool | **No** — before data exists, not a post-hoc audit |

**Tier decision: Tier-2 — "a predictability characterization of the Marson CD4 dataset."** Three of four are
clean non-occupants; all four agents independently returned *occupies_lane = false*. But **CIPHER** maps a
per-dataset structural quantity to predictability (verbatim: *"datasets with relatively high participation
ratios correspondingly exhibited lower R2 values (correlation R2 = 0.75; slope = -6.5)"*), which is close
enough to dataset-difficulty profiling that a **"first"** claim is not safely defensible — and scPerturBench's
paywalled full text cannot be fully excluded. The conservative rule sends this to **Tier-2**. *(A Tier-1
reading is arguable — CIPHER's participation ratio is a complexity/dimensionality construct, not an
irreducible-noise ceiling — but we do not overclaim on incomplete reads.)*

**Weak vs strong occupation — the pivot for the promote call: WEAK.** Verified against CIPHER's full text
(Kuznets-Speck et al.; bioRxiv 2025.06.27.661814 / Research Square 10.21203/rs.3.rs-7304871, PMC12363937),
**CIPHER does *not* define a model-agnostic predictability ceiling.** Its one cross-dataset result —
*verbatim:* "datasets with relatively high participation ratios correspondingly exhibited lower R2 values
(correlation R2 = 0.75; slope = −6.5)" — is a property of **its own linear method's** performance, not an
irreducible noise floor that bounds any model. scPerturBench (10.1038/s41592-025-02980-0) ranks 27 models
with **no** per-dataset ceiling and factors in only because its full text is paywalled. **No existing
single-cell predictability-*ceiling* method was found.** So this is weak occupation: a reliability-ceiling-
calibrated, positive-control-anchored, **seven-orthogonal-probe** scorecard is a clear step beyond a single
structural↔performance correlation — the contribution is **real at Tier-2** (Tier-1 is defensible), just not
safely labeled "first."

> Note: **CIPHER is the fluctuation/response-theory paper that probe P2 tested.** CIPHER predicts from the
> *covariance* (2nd moment); our P2 asked whether the *third* moment adds anything beyond covariance on real
> data → **negative** (0/24 strata). So this audit is adjacent to and consistent with CIPHER, not a competitor.

## G-PA.1 — the scorecard reproduces the committed verdicts: **PASS (faithful audit)**
`predictability_audit/` packages the seven probes + budget + do-operator control behind one
`run_audit("marson")` entry point. It is **stdlib-only** (no pandas), **does not retrain**, and **never
imports or modifies the frozen `core.eval`** — it reads the committed gate CSVs (ground truth), re-derives
each verdict from that probe's score + null + floor, and **self-checks against the committed verdict**.

**Result: 7/7 probes reproduced faithfully; the do-operator C2 control registers POSITIVE.** Packaging did
not change any answer. (Anti-triviality: no cell is reproducible by a trivially smoother/higher-SNR
reference — P1 is scored against a correlation null, P7 against its non-causal twin, the rest against
degree/label-preserving permutation nulls.)

### The Marson scorecard (`results/predictability_audit_gate.csv`)

![Marson CD4 Perturb-seq predictability scorecard](figures/predictability_scorecard.svg)

| Probe | Question | Verified reading | Verdict |
|---|---|---|---|
| **P1 Causal-matrix (Â_C)** | Does an explicit per-context causal matrix beat correlation under P≪G? | AUROC **0.62 < correlation-null 0.83** (oracle 1.00) | **FAIL** |
| **P2 Fluctuation (3rd-moment)** | Does the response 3rd moment predict what covariance cannot? | ΔR² ≈ **2.8e-6**, **0/24** strata significant | **NEGATIVE** |
| **P3 Single-cell SNR** | Would single-cell depth lift the pointwise floor? | x-donor **0.033**; **16%** cytokine genes SNR>3; gate RED | **NOISE-FLOOR** |
| **P4 Trajectory-geometry** | Is recoverability a trajectory-geometry artifact? | partial ρ **0.007 / 0.034** (p=0.75 / 0.55) | **NEGATIVE** |
| **P5 Donor-structure** | Does donor-conditioning beat donor-averaging? | conditioning **0.016 < averaging 0.034** | **NO-GO** |
| **P6 Relational-object** | Does any relational object reach 0.30? | best specific object **0.11 < 0.30** | **FAIL** |
| **P7 External causal-edge** | Is the edge recovery causal, or predictive? | causal **0.559 > null 0.500** but **= twin 0.569 (Δ−0.010)** | **IN-DISTRIBUTION** |
| **Budget (ceiling+floor)** | How much is recoverable; where does linear collapse? | floor **0.032/0.049** (perm_p 0); Ridge **0.02** vs do-op **0.56** of ceiling (gene) | floor real, small |
| **C2 Do-operator control** | Does the null machinery ever register a positive? | **+0.118 cond / +0.162 gene** vs twin | **POSITIVE** |

## The finding (the reframe)
Under honest measurement — every probe scored against its own degree/label-preserving null and read relative
to the measured reliability ceiling — **the recoverable signal is far narrower than the raw genome-scale
volume suggests.** Six probes sit at the noise floor or below a trivial reference; the **one accuracy
positive (C2) is in-distribution, not causal** (P7: it does not transfer to held-out external causal edges).

**The positive-control argument (why this is a map, not a failure):** the *same* null machinery that flags
six negatives + one in-distribution result still registers the do-operator C2 as a clear **positive**. So a
null cell means **"no signal here," not "no sensitivity"** — the instrument detects signal when signal
exists. That is what turns a pile of negatives into a *calibrated predictability map*.

## G-PA.2 — second-dataset port (Schmidt 2022): **the machinery ports (qualified); the floor finding is NOT re-tested**
*Appendix, subordinate to the Marson scorecard above — which is unchanged. This is a modest, honestly-bounded
upgrade: the audit **machinery** ports to a second primary-cell Perturb-seq dataset. It is **not** a claim that
the narrow-recoverable-signal finding replicates.*

**What ported.** The identical audit machinery — R1 per-perturbation reproducibility, R2 reliability ceiling,
R3 relational structure — was run **with no retrain** on **Schmidt et al. 2022** (GEO GSE190604; primary human
T cells, CRISPRa Perturb-seq, 73 selected screen-hit genes), on **Schmidt's own** recomputed floor. All three
probes reproduce **above their permutation floor** (`p ≤ 1/501`, B=500 — reported as the permutation floor, not
a precise p):

| Probe (Schmidt's own floor) | nostim | stim | vs Schmidt's own null |
|---|---|---|---|
| R1 reproducibility (cross-**well**) | 0.713 | 0.754 | null ≈ 0, p ≤ 1/501 |
| R2 reliability ceiling (split-half over cells, Spearman-Brown) | 0.925 | 0.933 | — |
| R3 relational-object S (target×target, cross-well split) | 0.863 | 0.926 | null 0.10 / 0.04, p ≤ 1/501 |

**The upgrade:** *"the instrument runs on a second primary-cell Perturb-seq dataset"* — **not** *"the finding
replicates."*

**Four bounds (load-bearing, not softened):**
- **(a) cross-well ≠ cross-donor — the decisive one.** R1 is **cross-well (technical replicate)**
  reproducibility. Marson's **0.03 floor is cross-donor (biological)**. Schmidt's public form has **no donor
  demux**, so the **cross-donor floor was NOT re-tested**. `0.71` and `0.03` are on **different axes** — not a
  comparison, and no evidence the floor finding generalizes.
- **(b) same lab** — Schmidt and our Marson dataset are the same consortium (Marson); **not** an independent group.
- **(c) CRISPRa vs our CRISPRi** — a different perturbation modality (gain- vs loss-of-function).
- **(d) 3 of 7 probes** — P4 (trajectory) and P5 (donor) are **N/A by construction on this data form**
  (2 pooled states; no donor labels); P7 (external causal) is deferred (needs a do-operator retrain).

**The limitation as the next step:** **floor-generalization is untested.** The experiment that would test it is
a **donor-demuxed second dataset** (so cross-donor reproducibility is computable) — a concrete, deferred next
step, not a hidden gap.

*(This is additive: the Marson seven-probe scorecard, the six-negatives finding, and the C2 positive control
above are unchanged. Full method + provenance: `GPA2_PORT.md`; numbers: `results/gpa2_scorecard.csv`.)*

## Honest ceilings
1. **Evaluation/methods contribution, not a new predictor.** Wins on dataset/causal-validity assessment and
   honesty of the negative-space map — the axis the field asked for — not on accuracy.
2. **Generalization is the real risk.** Demonstrated on **n=1** dataset ⇒ a case study. The likely cell-line
   second datasets cannot fully provide a clean port (P4/P5 degenerate). Do not overclaim "instrument."
3. **Cite the paradigm, claim the instantiation.** Intrinsic predictability is from ecology/forecasting; we
   instantiate it for Perturb-seq. Not "ImageNet/GLUE for cells."
4. **Every external number verified or flagged.** P7's 6,122 Freimer KO-DE + 45 Weinstock **direct** edges
   come from the C-FUSE report (`fusion-gates`, `882e12c`), kept separate from any Freimer full-text read;
   **Weinstock = Joshua S. Weinstock (PMC11605694), NOT "emdann"** (Emma Dann, a different researcher). C2
   verified against `results/benchmark_table.csv` (+0.118/+0.162); the floor 0.03 against
   `results/budget_cross_donor.csv` (0.032/0.049).

## Corrections where the repo/papers overrode the brief (they win)
- **`runs/<model>_<split>.parquet` are not committed** (they lived in the clobbered `DATA_ROOT`) — the
  scorecard reproduces from the committed **CSVs**, not raw run parquets.
- **No `core/residual.py|trajectory.py|donor.py|relational.py|fusion.py`** — the probes are **scripts**; their
  verdicts are committed CSVs. `predictability_audit/` packages those CSV verdicts, not nonexistent modules.
- **Novelty is Tier-2, not Tier-1** — CIPHER's per-dataset participation-ratio↔R² result and scPerturBench's
  paywalled text make "first" unsafe.

---

## The decision (resolved): **v2 promoted**

v2 was **promoted** and released as the submission of record. Rationale: it is a **strictly better-framed
submission of the same validated content** — it turns seven results + a positive control into one coherent,
novel-at-Tier-2 contribution (a *predictability scorecard*) on the exact axis the field's own papers ask for
("Virtual Cells as Causal World Models" calls for causal-validity metrics; nobody had built the dataset
audit). G-PA.1 is faithful; the second-dataset port (§ G-PA.2) shows the machinery ports (not floor
generalization); v1 remains the frozen, reversible fallback.

**State:** PR #12 merged to `main`; release `submission-v2` @ `a8878d5` is the submission of record; the
frozen tag `submission-fallback-v1` / `6476670` and CP2 are byte-untouched.
