# CP2 Results — the 2×2 (real GSE278572, NVIDIA L4)

Developer 2's half: JEPA pretraining (§7e) + the JEPA cells of the 2×2 + VOI + figures.
**Both hold-outs are now clean** (a gene-hold-out leak in the first run was found by an
adversarial audit, fixed, and the whole JEPA lane re-run gene-clean — see §7). Written
honest-first: the numbers, the pre-registered detector flags, and the run-to-run noise
are all surfaced.

## The 2×2 — condition hold-out (PRIMARY, zero-shot Stim48hr), Pearson-δ on top-50 DEGs

| | mask **off** | mask **on** |
|---|---|---|
| **random-init** | 0.2255 (noncausal) | 0.3436 (causal) |
| **JEPA-init** | 0.2387 (jepa_only) | 0.3404 (jepa_causal) |

- **C2 (do-operator, mask on − off): +0.1099** (random +0.118, jepa +0.102). The corrected
  do-mask improves **both** Pearson-δ (0.226→0.344) **and** perturbation-discrimination
  (perturbench_rank 0.483→0.457, lower better) — a multi-axis inductive-bias signal, and
  pre-registered to report regardless of leaderboard position.
- **C3 (JEPA-init, jepa − random): +0.0050** (off +0.013, on −0.003) — **null.** (The first,
  gene-leaky run gave −0.002; the ~0.01 shift is run-to-run JEPA-pretraining variation — i.e.
  C3 sits inside the noise band either way.) Pre-registered corroboration of Cell-JEPA
  ("JEPA improves absolute-state reconstruction, not effect-size/delta").

### ⚠️ Essential caveat (pre-registered mode-collapse detector)
**All four transformer cells are red-flagged** (perturbench_rank > 0.4: causal 0.457,
noncausal 0.483, jepa_causal 0.459, jepa_only 0.474). **Ridge is the only non-collapsed
model (0.365) AND the top Pearson-δ scorer (0.384 > 0.344).** Honest reading: *the do-mask
improves a model that still fails perturbation discrimination*; ridge's function/network
priors win raw accuracy. This is the pre-registered "priors win accuracy, the do-operator
isolates the intervention" story, told straight.

## Gene hold-out (secondary, now hold-out-clean), Pearson-δ

| | mask off | mask on |
|---|---|---|
| random-init | 0.2056 (noncausal) | 0.3675 (causal) |
| JEPA-init | 0.2483 (jepa_only) | 0.3609 (jepa_causal) |

- **C3 (gene): off +0.043 (jepa_only 0.248 vs noncausal 0.206), on −0.007** (jepa_causal 0.361
  vs causal 0.368); mask-averaged +0.018.
- The gene-clean re-run **shrank the mask-off effect from the leaky +0.057 to +0.043** — so
  part of the original apparent benefit was leakage, but a **modest clean signal survives** for
  the direct-regression model. It is **mask-dependent** (mask-on reverses) and **both cited
  cells are mode-collapse-flagged**, so treat it as a small, exploratory positive — **not** a
  robust "JEPA helps unseen-gene interpolation" claim. Consistent with the pre-registered
  expectation of little-to-no JEPA delta benefit.

## S1 — Value of Information (VOI)
Ensemble-disagreement VOI ranks which perturbations are most worth measuring. Sample-efficiency
(Ridge surrogate on the condition hold-out; full-screen Pearson-δ = 0.384): **VOI-guided
selection reaches 90% of full-screen at 75.5% of perturbations vs random's 87.4%** — VOI beats
random, modestly.

## Bottom line
- **C2 (do-operator) is the real, robust effect** (+0.11), with the honest caveat that the
  transformers are mode-collapsed and ridge wins raw accuracy.
- **C3 (JEPA-init) is null-to-small on the delta task** (all effects within ~±0.02, the
  run-to-run noise band) — the pre-registered Cell-JEPA corroboration.
- **S1 (VOI) works** — disagreement-guided selection is more sample-efficient than random.

## Challenger (rejected): can the transformers be made to discriminate? — v2

We probed the one soft spot (all four transformer cells fail the mode-collapse detector) with a
gated **perturbation-discrimination InfoNCE** term added to the loss (batch-centered cosine,
in-batch negatives, same-gene masked) — same trainer, split, epochs=40, JEPA checkpoint; only the
objective changes. `core.models.discrim_v2`, weight tuned on a **disjoint 15% dev split**.

**Diagnostic first (residual PerturBench rank):** the collapse is *real*, not a metric artifact —
removing the shared activation-state profile leaves the transformer rank at ~0.46 (barely moved),
while ridge stays at ~0.36. So a fix (not just a reframe) was warranted.

**Result (`results/benchmark_table_v2.csv`, w=0.3):** the term improved discrimination for the
mask-**on** cells (causal gene rank 0.440→**0.364**, *clears*; causal condition 0.457→0.407) and
even **grew the C2 do-operator gap** (condition +0.118→+0.180, gene +0.162→+0.194, because the
mask-off models collapse harder). BUT: (1) it did **not** clear the flag on the primary **condition**
hold-out (causal 0.407, jepa_causal 0.427 still > 0.4); (2) it didn't help the mask-off cells; and
(3) it **cratered absolute accuracy** (causal condition 0.344→0.248; mask-off cells to ~0.05).

**Verdict (pre-committed §4 gate): REJECTED.** Promotion required causal *and* jepa_causal to clear
rank < 0.4 on the reported hold-outs *and* the C2 gap to hold ≥ +0.10. The C2 gap held, but
discrimination did not clear on the condition hold-out — and the accuracy cost is steep. **CP2
stands as the submission of record.** The finding is itself real and reportable: *there is a genuine
accuracy↔discrimination trade-off — forcing discrimination via the loss buys only a partial rank
improvement at a large Pearson-δ cost, so the honest CP2 framing (accurate, do-operator effect
intact, flag surfaced) is retained.* v2 artifacts kept in `runs/*_v2.parquet` + `benchmark_table_v2.csv`.

## Figures
`figures/figure1_benchmark.png` (benchmark table, collapse in red), `figure2_2x2.png` (the 2×2,
C2/C3 read-out), `figure3_subsampling.png` (VOI vs random), `figure4_biology.png` (top VOI genes).

## JEPA pretraining (G4) — sanity
20,000 steps on ~1M single cells (D1 Rest+Stim8hr, 500k each, gene-hold-out-clean), loss
1.39→0.065, teacher-embedding std 0.010→0.15 (grew ~15×, **no collapse**), gate-projected
0.90 h / actual 0.93 h. The JEPA checkpoint transfers into `CausalCisTransFormer.encoder`
exactly (0 missing / 0 unexpected).

## Provenance
- All four 2×2 cells trained with `CausalConfig(epochs=40)` — **identical to CP1's causal/
  noncausal** (`scripts/run_cp1.py`), differing solely by the JEPA encoder init.
- JEPA cache is hold-out-clean on all three axes: excludes Stim48hr (condition), donor D4
  (donor probe), and the 1,729 gene-hold-out genes' cells (467k cells dropped by the
  `ingest_assigned_guide(holdout_genes=…)` filter).
- Split SHA `fd2b8c21…`; eval + split + features frozen (Dev 1); consumed unchanged.
- Two confounds were caught by adversarial audits and fixed before these numbers: an epochs
  mismatch (60→40) and the gene-hold-out cache leak (re-run gene-clean).
