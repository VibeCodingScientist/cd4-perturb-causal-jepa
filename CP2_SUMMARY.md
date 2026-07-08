# CP2 — Team Summary (Developer 2: JEPA + analysis)

**Status: CP2 complete and merged to `main`** (the JEPA cells of the 2×2, VOI, and the
demo figures). Trained on **real GSE278572** data on the shared NVIDIA L4. A
hold-out-clean gene-split re-run is finishing (see §6); the primary (condition)
result below is final. Written honest-first — every pre-registered detector flag and
confound is surfaced.

Read alongside: [`HANDOFF.md`](HANDOFF.md) (CP1), [`RESULTS.md`](RESULTS.md) (detail),
[`hypotheses.md`](hypotheses.md) (pre-registration), [`UNIFIED_BUILD_PLAN.md`](UNIFIED_BUILD_PLAN.md).

---

## 1. TL;DR

- The experimental core is a **2×2 ablation** (encoder-init × causal-mask). CP1 delivered
  the random-init row (causal/noncausal) + baselines; **CP2 delivers the JEPA row**
  (`jepa_only`, `jepa_causal`) + VOI + figures.
- **C2 (do-operator) is the headline and it holds:** the corrected do-mask adds **+0.12**
  Pearson-δ on the zero-shot condition hold-out — and improves *both* accuracy and
  perturbation-discrimination. **Caveat (surfaced): all four transformer cells are
  mode-collapse-flagged; ridge (non-collapsed) wins raw accuracy.**
- **C3 (JEPA-init) is a clean null on the condition hold-out** — a *pre-registered*
  corroboration of Cell-JEPA ("JEPA helps absolute-state, not effect-size delta").
- **S1 (VOI):** ensemble-disagreement VOI beats random selection (90% of full-screen at
  75.6% of perturbations vs 87.4%).
- JEPA pretraining ran correctly (1M single cells, EMA teacher, **no collapse**) and its
  checkpoint initializes the causal encoder **exactly** (0 missing / 0 unexpected).

---

## 2. The 2×2 — condition hold-out (PRIMARY, zero-shot Stim48hr, hold-out-clean)

Pearson-δ on top-50 DEGs (higher better):

| | mask **off** | mask **on** |
|---|---|---|
| **random-init** | 0.2255 (noncausal) | 0.3436 (causal) |
| **JEPA-init** | 0.2211 (jepa_only) | 0.3440 (jepa_causal) |

- **C2 = +0.1205** (mask on − off; random +0.118, jepa +0.123). The do-mask improves both
  Pearson-δ (0.226→0.344) **and** perturbation-discrimination (perturbench_rank
  0.483→0.457, lower better). Pre-registered to report regardless of leaderboard.
- **C3 = −0.0020** (jepa − random; off −0.004, on +0.0004). **Null** — pre-registered
  corroboration of Cell-JEPA.

### ⚠️ Essential caveat — the pre-registered mode-collapse detector
All four transformer cells exceed the 0.4 discrimination threshold (perturbench_rank:
causal 0.457, noncausal 0.483, jepa_causal 0.460, jepa_only 0.482 → all flagged red).
**Ridge is the only non-collapsed model (0.365) and also the Pearson-δ winner (0.384 >
0.344).** Honest reading: *the do-mask improves a model that still fails perturbation
discrimination*; ridge's function/network priors win raw accuracy. This is exactly the
pre-registered "priors win accuracy, the do-operator isolates the intervention" story.

## 3. Gene hold-out (secondary) — clean re-run in flight

| | mask off | mask on |
|---|---|---|
| random-init | 0.2056 (noncausal) | 0.3675 (causal) |
| JEPA-init | 0.2631\* (jepa_only) | 0.3613\* (jepa_causal) |

\* The gene-split JEPA numbers in the first run were **confounded** — the JEPA cache
excluded Stim48hr + donor D4 but not held-out *genes*, so pretraining saw the held-out
genes' knockdown cells. The mask-off "+0.057" was inflated and mask-dependent (mask-on
reverses, −0.006). **Fixed in code** (`ingest_assigned_guide(holdout_genes=…)`); a
hold-out-clean re-run is finishing now (§6) and is expected to show the pre-registered
**null**. The condition hold-out (§2) is clean and unaffected.

## 4. Claims scorecard (vs `hypotheses.md`)

| Claim | Outcome |
|---|---|
| **C1** (causal ≥ baselines incl. TabPFN, condition) | Mixed/honest: ridge 0.384 > causal 0.344 on condition; causal dominates gene hold-out (0.368 vs ridge 0.019). TabPFN N/A (license-gated). |
| **C2** (causal > non-causal, do-operator) | **Confirmed** (+0.12, multi-axis) — with the mode-collapse caveat above. Headline. |
| **C3** (JEPA-init helps condition) | **Null** on condition (pre-registered Cell-JEPA corroboration). Gene: clean re-run pending. |
| **S1** (VOI ranks worth-measuring perturbations) | Supported: VOI-guided 75.6% vs random 87.4% to reach 90% of full-screen. |

## 5. Deliverables on `main`

- `core/models/jepa.py` — Cell-JEPA (EMA teacher, single-cell value-masking, cosine +
  recon, VICReg collapse guard, measure-then-extrapolate gate). Queue entry `run_jepa`.
- `core/models/jepa_integration.py` — JEPA→causal weight transfer + G5 fine-tune
  (`run_jepa_finetune`; replicates `cc._run` with the encoder init inserted).
- `core/models/jepa_data.py` — single-cell ingestion (holdout-clean, mmap `.npy` cache).
- `core/voi.py`, `core/ablation.py`, `figures/make_figures.py`, `scripts/fetch_jepa_cells.py`,
  `scripts/cp2_finalize.py`.
- `Snakefile` — CP2 rules wired (`jepa`, `jepa_finetune`, `jepa_cells`, `cp2`).
- `results/benchmark_table.csv` (all 12 rows), `figures/figure{1,2,3,4}.png`,
  [`RESULTS.md`](RESULTS.md). **~60 tests green.**

## 6. What's running now — clean gene-holdout re-run

Chained on the box (`~/cd4-ws2/rerun_clean.sh`): clear confounded cache → re-fetch D1
Rest+Stim8hr with the gene filter (excludes the 1,729 held-out genes' cells) → G4
re-pretrain → G5 re-fine-tune → `cp2_finalize`. ~3 h. On completion I'll update the
gene-holdout row in `RESULTS.md` + figures + benchmark and push. Old cache kept as
`cells_confounded` for comparison.

## 7. Reproduction

```bash
# baselines + causal (CP1):   see RUNBOOK.md
# JEPA cells (CP2):
python scripts/fetch_jepa_cells.py --donors D1 D2 D3 --hvg-path split/hvg_3000.txt  # flagged: ~S3 cells
python gpu_queue.py submit jepa            # G4 pretrain -> checkpoints/jepa_final.pt
python gpu_queue.py submit jepa_finetune   # G5 -> runs/jepa_{only,causal}_*
python scripts/cp2_finalize.py             # 2x2 + VOI + figures
# or: snakemake --cores all cp2
```

## 8. Coordination notes (for Developer 1)

1. **G5 encoder-init hook (nice-to-have).** `finetune_jepa_models` currently *replicates*
   `cc._run` with `load_jepa_into_encoder(model.encoder, ckpt)` inserted (verified
   faithful, epochs=40 to match CP1). A 2-line `encoder_init_ckpt=None` param on
   `cc._run` would let it call the public runner directly — `finetune_jepa_models`
   auto-detects the hook.
- **JEPA pretraining sanity:** 20k steps on 1M cells, loss 1.39→0.065, teacher-embedding
  std 0.010→0.15 (grew ~15×, no collapse), gate-projected 0.90 h / actual 0.91 h. Weight
  transfer into `CausalCisTransFormer.encoder`: **0 missing / 0 unexpected**.
- **Config match:** jepa_causal/jepa_only use `CausalConfig(epochs=40)` — identical to
  CP1's causal/noncausal (`run_cp1.py`), differing solely by the JEPA init (an earlier
  epochs=60 confound was caught + fixed before any reported number).

## 9. Open items

- Clean gene-holdout re-run (finishing) → final gene number.
- TabPFN still N/A (license-gated) — the one CP1 model missing.
- Transformer discrimination is borderline (rank 0.44–0.48); worth probing whether a
  sharper prediction setup lifts it below 0.4.
- Context prior is an ESM-2 PCA stand-in (node2vec-over-STRING wired but not run).

## 10. Ops

Box `ubuntu@54.163.21.62` (L4, 1 TB disk) — kept running for the re-run. Idle otherwise;
billed hourly. GPU budget for CP2: ~1 overnight-equivalent (G4 55 min + G5 67 min +
re-run). Artifacts (benchmark, figures) mirrored to git.
