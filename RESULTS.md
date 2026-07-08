# CP2 Results — the 2×2 (real GSE278572, NVIDIA L4)

Developer 2's half: JEPA pretraining (§7e) + the JEPA cells of the 2×2 + VOI + figures.
Written to be **honest first** — the numbers, the pre-registered detector flags, and the
confounds are all surfaced (a 4-dimension adversarial audit corrected two overclaims in
an earlier draft; those corrections are baked in here).

## The 2×2 — condition hold-out (zero-shot Stim48hr), Pearson-δ on top-50 DEGs

| | mask **off** | mask **on** |
|---|---|---|
| **random-init** | 0.2255 (noncausal) | 0.3436 (causal) |
| **JEPA-init** | 0.2211 (jepa_only) | 0.3440 (jepa_causal) |

- **C2 (do-operator, mask on − off): +0.1205** (random +0.118, jepa +0.123). The corrected
  do-mask improves **both** Pearson-δ (0.226→0.344) **and** perturbation-discrimination
  (perturbench_rank 0.483→0.457, lower = better) — a multi-axis inductive-bias signal, and
  it's pre-registered to be reported regardless of leaderboard position.
- **C3 (JEPA-init, jepa − random): −0.0020** (off −0.004, on +0.0004) — **null** on the
  condition hold-out. This is a pre-registered outcome: it **corroborates Cell-JEPA**, which
  found JEPA improves absolute-state reconstruction but **not** effect-size/delta estimation.
  This split's JEPA cache is hold-out-clean (excludes Stim48hr + donor D4).

### ⚠️ Essential caveat (pre-registered mode-collapse detector)
**All four transformer cells are red-flagged** by the mode-collapse detector
(perturbench_rank > 0.4: causal 0.457, noncausal 0.483, jepa_causal 0.460, jepa_only 0.482).
**Ridge is the only non-collapsed model (0.365) AND the top Pearson-δ scorer (0.384 > 0.344).**
So the honest reading of C2 is: *the do-mask improves a model that still fails perturbation
discrimination.* The transformers score moderate Pearson-δ largely via shared activation-state
DEGs; ridge (function/network priors, no collapse) wins raw accuracy. This is the pre-registered
"tabular/prior model wins accuracy; the do-operator isolates the intervention" story, told
straight — not a "JEPA/causal beats everything" claim.

## Gene hold-out (secondary) — ⚠️ CONFOUNDED for the JEPA cells

| | mask off | mask on |
|---|---|---|
| random-init | 0.2056 (noncausal) | 0.3675 (causal) |
| JEPA-init | 0.2631 (jepa_only) | 0.3613 (jepa_causal) |

The gene-hold-out JEPA numbers are **not hold-out-clean**: the JEPA cache filtered by
condition and donor but (in the run that produced these numbers) **not by gene**, so cells
perturbed for held-out genes were in pretraining — the encoder saw the held-out genes'
knockdown phenotypes. The mask-off cell (+0.057) is therefore confounded, and the effect is
**mask-dependent** (mask-on reverses: −0.006; mask-averaged +0.026). Both cited cells are also
mode-collapsed. Treat as a **non-robust, exploratory** observation; a clean run would most
likely show the pre-registered **null** (JEPA does not help gene-hold-out delta).

**Fixed in code (not re-run):** `ingest_assigned_guide(..., holdout_genes=...)` now drops cells
whose `perturbed_gene_id` is in the split's `gene_holdout`, and `fetch_jepa_cells.py` threads it
through. A clean gene-hold-out result requires rebuilding the cache + re-pretraining (≈3 h GPU +
~277 GB re-download); deferred as a documented follow-up since the **primary (condition) result
is clean** and the pre-registration already predicted a gene-hold-out null.

## S1 — Value of Information (VOI)
Ensemble-disagreement VOI ranks which perturbations are most worth measuring. Sample-efficiency
(Ridge surrogate on the condition hold-out, full-screen Pearson-δ = 0.384): **VOI-guided
selection reaches 90% of full-screen at 75.6% of perturbations vs random's 87.4%** — VOI beats
random, modestly.

## Figures
`figures/figure1_benchmark.png` (benchmark table, collapse in red), `figure2_2x2.png` (the 2×2,
C2/C3 read-out), `figure3_subsampling.png` (VOI vs random), `figure4_biology.png` (top VOI genes).

## JEPA pretraining (G4) — sanity
20,000 steps on ~1M single cells (D1 Rest+Stim8hr, 500k each), loss 1.39→0.065, teacher-embedding
std 0.010→0.15 (grew ~15×, **no collapse**), gate-projected 0.90 h / actual 0.91 h. The JEPA
checkpoint transfers into `CausalCisTransFormer.encoder` exactly (0 missing / 0 unexpected).

## Provenance
- jepa_causal/jepa_only trained with `CausalConfig(epochs=40)` — **identical to CP1's causal/
  noncausal** (`scripts/run_cp1.py`), differing solely by the JEPA encoder init (an earlier
  epochs=60 confound was caught and corrected before these numbers).
- Split SHA `fd2b8c21…`; eval + split + features frozen (Dev 1); consumed unchanged.
