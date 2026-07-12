# G-PA.2 — porting the predictability audit to a second dataset (branch `gpa2-port`)

*Developer 2, autonomous, AFK-safe. The one deferred experiment that could upgrade the v2 claim from "an
audit demonstrated on n=1" to **"audit machinery that ports to a second dataset"** — generalization of the
floor *finding* itself stays out of scope here (the cheap public form carries no cross-donor labels). Runs on
this branch; both releases (`submission-v2` / `a8878d5`, `submission-fallback-v1` / `6476670`) stay
byte-untouched. No do-operator retrain. The one decision left for the lead — fold the port into v2 or leave
it as a branch result — is not made here.*

## Step 0 — dataset triage → **Track A dataset (with a same-lab caveat + a form limitation)**
Surveyed the public Perturb-seq landscape for a cheaply-ingestible (<~2 GB, open) second dataset (retrieved
2026-07-12; a 4-agent survey, sizes/access verified against GEO/Zenodo/Figshare):

| Dataset | System | Trajectory | Modality | n_pert | Cheapest open form | Track |
|---|---|---|---|---|---|---|
| **Schmidt 2022 (GSE190604)** — *chosen* | **primary human T, 2 donors** | **yes (rest↔restim)** | CRISPRa Perturb-seq | 73 (screen hits) | GEO mtx bundle **~0.95 GB** | **A** |
| Replogle 2022 / CausalBench (K562, RPE1) | cell line | no | CRISPRi | ~600–2400 curated | Figshare pseudobulk 0.08–0.10 GB; scPerturb 1.2–1.5 GB; raw 65.8 GB | B |
| Norman 2019 / Adamson 2016 | cell line (K562) | no | CRISPRi | ~230 / ~80 | scPerturb h5ad 35–471 MB | B |
| ShifrutMarson 2018 | primary T, multi-donor | **unconfirmed** | CRISPR-KO | unconfirmed | scPerturb h5ad 872 MB | A? |

**Chosen: Schmidt et al. 2022, Science** ("CRISPR activation and interference screens decode stimulation
responses in primary human T cells"; DOI 10.1126/science.abj4008; PMC9307090; GEO **GSE190604**). It is the
only cheaply-ingestible dataset whose underlying *study* carries all three Track-A properties — primary human
T cells, **2 donors**, and a **rest↔restimulation** axis. **Caveat carried into the results:** the cheapest
open GEO form (aggregated mtx bundle) carries **no per-cell donor-demux labels and only 2 pooled activation
states (no trajectory geometry)**, so **P4 (trajectory) and P5 (donor) are N/A from this form** — only
cross-well (replicate) structure is recoverable (see BOUND 1). Schmidt was chosen as the closest *cheap*
approach to the Track-A ideal, **not** because the cross-donor floor could actually be ported.

**Two honest caveats carried into every claim:**
1. **Same lab (Marson), not independent.** Schmidt is the *direct predecessor* of the genome-scale CD4⁺
   Perturb-seq the audit validates → same-consortium, cross-platform/modality, **not** lab-independent.
2. **Different perturbation modality + scale.** Schmidt is **CRISPRa** (gain-of-function) with only **73**
   selected screen-hit perturbations; Marson is CRISPRi (loss) at genome scale.

*(Rejected: cell-line datasets (Replogle/Norman/Adamson) → Track B, a modest footnote with P4/P5 N/A by
construction; ShifrutMarson trajectory unconfirmed. No genuinely lab-independent primary/multi-donor/
trajectory Perturb-seq ingests cheaply — Track A is only achievable same-consortium, and even then donor
labels don't survive in the cheap form.)*

## Pre-registered expectations (stated BEFORE computing any probe)
The instrument test is "does the *identical* audit produce a **coherent, interpretable, null-discriminating**
scorecard on Schmidt, with the probes that Schmidt's *own* nulls can adjudicate behaving sensibly." Because
Schmidt's 73 perturbations are *selected strong CRISPRa hits*:

- **E1 — reproducibility (P3/budget):** the per-perturbation **cross-well (replicate)** reproducibility should
  be detectable and high (the cross-*donor* floor is **not computable** from the cheap form — no donor labels
  — so only the replicate axis is pre-registered; any comparison to Marson's cross-donor 0.03 is a
  different-axis comparison and is **not** a pre-registered success condition).
- **E2 — reliability ceiling:** computable and sensible (0 < ceiling < 1) via split-half over cells.
- **E4 — relational-object (P6):** a specific-space object should **exceed Schmidt's own degree-preserving
  null**; if it also exceeds Marson's best relational object (loadings ~0.11) or the 0.30 pre-registered bar,
  that is an interpretable *difference* (strong-hit datasets carry more relational structure), not incoherence.
- **E5 — donor-structure (P5):** *contingent on donor labels surviving in the cheap GEO form — they do not, so
  E5 is untestable here (P5 N/A).*
- **E6 — trajectory (P4):** only 2 pooled states, no geometry → **N/A** from this form.
- **P1/P2 (mechanism), P7 (external):** deferred (heavy / needs a retrain). Marked N/A, not run.

**Instrument criterion (pre-registered):** COHERENT if (a) the computable probes yield interpretable numbers
on Schmidt's *own* recomputed floor/ceiling (never Marson's), (b) the differences that Schmidt's own nulls can
adjudicate behave sensibly (ridge/relational exceed Schmidt's own nulls) — **excluding E1's "floor above
Marson's 0.03," a cross-donor-vs-cross-well comparison that was never testable and is NOT counted as
confirmation**, and (c) the null machinery discriminates signal from no-signal. Coherent → **audit machinery
ports (qualified)**; degenerate → **case study, n=1**.

## Fences honored
Same probe *definitions* + nulls + ceiling logic as the Marson audit — **only the ingestion changes**;
Schmidt's floor/ceiling are **recomputed on Schmidt** (never imported); no do-operator retrain; ingestion
writes to an isolated `~/gpa2-data`, never `DATA_ROOT`; both releases + CP2 + `core.eval` byte-untouched; no
release cut, no merge.

---

## Computed port — Schmidt 2022 GSE190604 (results)
**Ingestion verified (Stage 1–2a), not assumed:** the 8 mtx lanes are the `mRNA-{nostim,stim}-well{1-4}`
samples; lane→condition confirmed **empirically by activation markers** (lanes 1-4 nostim ACT≈0.25, lanes 5-8
stim ACT≈0.95, clean 4/4 gap 0.70 — IFNG 0.16→1.4, IL2RA 0.3→2.3, GZMB 0.3→2.5); control = **`NO-TARGET`**
(4,000 cells); **73 perturbed genes** (~70 per probe after per-condition singlet-guide filtering: n_targets
70/71/69), 61,041 singlet-guide cells, 4 wells/condition, 2,000 HVG computed on Schmidt itself.

> **Read BOUND 1 first:** R1 is cross-**WELL** (technical replicate); Marson's 0.03 is cross-**DONOR**
> (biological). Different axes — the Marson column is context, **NOT** a comparison.

| Probe (Schmidt's own floor) | nostim | stim | vs Schmidt's own null | Marson (DIFFERENT AXIS — not comparable) |
|---|---|---|---|---|
| **R1** per-perturbation reproducibility (cross-**well**) | 0.713 | 0.754 | null ≈0, p ≤ 1/501 | cross-**donor** 0.03 — different (biological) axis; **not** comparable |
| **R2** reliability ceiling (split-half over cells, Spearman-Brown) | 0.925 | 0.933 | — | — |
| **R3** relational-object S (target×target, cross-well split) | 0.863 | 0.926 | null 0.10 / 0.04, p ≤ 1/501 | specific S 0.008 (observed) — gap inflated by selected-strong-hit construction (BOUND 2) |

*All p-values are the **permutation floor 1/501** (B=500 permutations; observed exceeded all 500) — the shared
value is the perm-budget resolution limit, not three coincident precise p-values.*

**N/A here:** P5 donor (no donor demux in the public form), P4 trajectory (2 pooled states, no geometry),
P1/P2 mechanism (heavy re-implementation), P7 external (needs a do-operator retrain — out of AFK scope).

## The honest read — the *machinery* ports; the *floor finding* is NOT tested
- **What IS demonstrated (positive):** the identical audit machinery (ingestion → probes → degree/label-
  preserving nulls) runs on a second, independently-processed primary-cell Perturb-seq dataset and produces a
  **coherent, null-discriminating scorecard** — every probe clears Schmidt's own null (p ≤ 1/501). The audit
  is a **reusable instrument**, not a one-off Marson script.
- **BOUND 1 — the decisive caveat: R1 is cross-WELL (technical replicate), not cross-DONOR (biological).**
  Marson's 0.03 floor was cross-*donor*. Schmidt's public form has **no donor demux**, so the cross-donor
  floor **could not be ported**. The Marson *floor finding was not tested* on Schmidt; 0.71 vs 0.03 is a
  different-axis non-comparison.
- **BOUND 2 — high by construction:** 73 *selected strong screen-hits* + on-target CRISPRa activation ⇒ high
  cross-well reproducibility is expected; not the *hard* test (recovering a floor in a genome-scale mix) that
  gave the Marson audit its value.
- **BOUND 3 — not independent:** same lab (Marson), CRISPRa vs our CRISPRi.
- **BOUND 4 — 3 of 7 probes:** P4/P5 N/A from the public form; P1/P2/P7 deferred.

## Routing — **COHERENT port → audit machinery ports (qualified); floor-generalization NOT shown**
The pre-registered coherence criterion is **met**: the audit produces a coherent, interpretable,
null-discriminating scorecard on the new data (on the probes Schmidt's own nulls can adjudicate; **E1's "floor
above Marson's 0.03" was not confirmed — the cross-donor floor was never computed**). So:
- **Earned:** *"the predictability-audit **machinery** ports to a second, same-consortium primary-cell CRISPRa
  Perturb-seq dataset — coherent, null-discriminating scorecard, 3 model-free probes on the dataset's own floor."*
- **NOT earned:** floor-*finding* generalization (cross-donor floor untested — no donor labels), independent-lab
  validation (same lab), or a full 7-probe instrument. This is a **machinery-portability demonstration**, not
  proof the narrow-recoverable-signal finding holds on a second dataset.

## The one decision for the lead (with a recommendation)
**Fold the port into v2, or leave it as a branch result?** Recommendation: **soft lean — fold in as a
clearly-caveated second-dataset appendix**, keeping the v2 *headline* unchanged. It genuinely upgrades "an
audit demonstrated on n=1" → "the audit machinery ports to a second primary-cell Perturb-seq dataset and
produces a coherent scorecard," strengthening the *reusable-instrument* framing — **only if all four bounds
are stated verbatim** (especially cross-well ≠ cross-donor). If the lead judges "machinery ports but the floor
finding was not re-tested" too thin to add, **leave it as this reviewable branch result.** **Not folded here.**

## Fences (verified)
Both releases byte-untouched (`submission-v2`/`a8878d5`, `submission-fallback-v1`/`6476670`); this branch
`gpa2-port` touches **0 lines** of `core.eval`/`benchmark_table.csv`; ingestion wrote only to the isolated
`~/gpa2-data` (never `DATA_ROOT`); no do-operator retrain; no release cut; nothing merged.

Deliverables: `GPA2_PORT.md`, `results/gpa2_scorecard.csv`, `scripts/gpa2_stage{1,2a,2b}*.py`.
