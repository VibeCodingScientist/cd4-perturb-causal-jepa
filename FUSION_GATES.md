# C-FUSE — External validation of the do-operator (Phase 1 overlap + Phase 1b powered test)

*Developer 2, autonomous. Does the do-operator's genome-scale causal signal replicate causal edges
measured in other (same-consortium) CD4 datasets — and is that recovery **causal-specific** (better than a
non-causal twin) or merely predictive? The frozen submission (`submission-fallback-v1` / `6476670`) and
CP2 (`benchmark_table.csv`) are byte-untouched; nothing is merged to `main`.*

**Outcome (two parts):**
- **G-F.1 (overlap):** PASS — 15,542 measurable cross-dataset edges (≫50).
- **G-F.2 (powered, non-circular, Phase 1b):** the do-operator recovers held-out external edge
  **directions above chance** (combined 0.559 vs null 0.500, cluster-robust across all **9 usable
  regulators**, TBX21 n=1 excluded, binom p=0.004) — **but its non-causal twin recovers them equally well
  or marginally better** (causal−twin = −0.010, regulator cluster-bootstrap 95% CI **[−0.013, −0.005],
  excludes 0**). **Verdict: the external-edge recovery is real but CORRELATIONAL, not causal-specific —
  G-F.2 FAILS the do-operator > twin bar.** The within-dataset C2 advantage (+0.118 cond / +0.162 gene)
  does **not** transfer to external causal structure. **A decisive, well-powered negative — the 7th on the
  frontier.**

## Source-of-truth corrections (repo/paper win over the brief)
- The brief's **"emdann 2024"** is actually **Weinstock et al. 2024, Cell Genomics** (LLCB). Corrected.
- The brief's **"211 edges over 84 TFs" is [VERIFIED] correct** — 350/211/151 at |β|≥0.020/0.025/0.030,
  matching the paper exactly (`data/external/PROVENANCE.md`).
- **Both external papers are Marson/Pritchard-consortium** — same lab as our GSE278572. So this is
  cross-**platform/modality** validation (our CRISPRi Perturb-seq vs their CRISPR-KO + LLCB / bulk
  RNA-seq), **not** fully independent-lab replication. Any "external validation" claim must say so.

## Step 0 — external edge tables obtained ✅ (both)
- **Weinstock Table S1** (`mmc2.xlsx`): 6,972 β-signed parent→child pairs among 84 TFs → **211** direct
  causal edges at |β|≥0.025. [VERIFIED]
- **Freimer Suppl. Data 1** (RNA_Seq_results.csv, 269,080 rows): **50,654** regulator→target DE edges at
  FDR<0.05 (logFC sign = direction), 24 regulators. [VERIFIED] (indirect DE, unlike Weinstock's direct
  edges — kept separate throughout.)
- URLs, dates, row counts, read-tiers in `data/external/PROVENANCE.md`.

## G-F.1 — measurable overlap: **PASS**
Measurable shared edge = external edge (regulator R → target T) where R is in our **assayed
perturbations** AND T is in our **HVG readout**.

| set | edges | measurable | measurable regulators |
|---|---|---|---|
| Weinstock (direct) | 211 | 104 | 22 |
| Freimer (DE) | 50,654 | 15,438 | 24 |
| **combined** | 50,865 | **15,542** | 28 |

**G-F.1 verdict: PASS** (15,542 ≫ 50). Of the 28 measurable regulators, **10 are also in the HVG panel**
(so the gene-token do-operator can represent them as perturbations) — these are the testable set for the
powered non-circular test below. (Phase 1's original blocker — only 2 regulators land in the *frozen* 15%
gene-holdout — is what motivated the Phase-1b re-scope.)

## C-FUSE 1b — powered, non-circular G-F.2 (regulator-holdout re-train, GPU)
Rather than rely on the frozen split's incidental 2-regulator holdout, we **purpose-built a
regulator-holdout**: remove the 10 HVG-resident external regulators from the training pseudobulk, **re-train
the do-operator and its non-causal twin from scratch**, and predict those 10 regulators **zero-shot**. This
is genuinely non-circular — the model never saw these regulators' perturbations in training.

**Method (all on the L4 box, frozen panel/split pinned, nothing frozen overwritten):**
- Held out **10 regulators** (in HVG): TBX21, GATA3, FOXP1, TNFAIP3, MYB, IRF1, STAT5A, KLF2, IL2RA, IRF4.
  Train perturbations 9,755 → 9,745 (−10). 11,438 training samples.
- Re-trained **causal** (do-mask, 26.1 min) + **non-causal twin** (20.7 min), identical recipe/seed;
  predicted Δ over 3,000 HVG for each held-out regulator (`results/fusion_pred_{causal,noncausal}.parquet`).
- **6,167 measurable held-out edges**: Freimer DE = 6,122 (9 regs), Weinstock direct = 45 (8 regs).
- **Metric:** sign/direction agreement (not effect-size). **Knockdown sign convention:** Freimer edge sign
  is already a KO logFC → expected = edge sign; Weinstock β is an activation coefficient → knockdown flips
  it → expected = −edge sign. **Null:** degree-preserving (each edge keeps its regulator; target resampled
  over the HVG panel). Because the 6,167 measurable held-out edges cluster into only 10 regulators (9 after
  dropping TBX21, n=1), all inference is also reported **regulator-level** (the honest independent unit) via
  a per-regulator sign test and a **cluster bootstrap over regulators**.

**Result (derived convention):**

| edge set | edges | regs | causal acc | twin acc | causal−twin | vs null (both) |
|---|---|---|---|---|---|---|
| Freimer (DE) | 6,122 | 9 | 0.560 | 0.569 | −0.009 | p = 5×10⁻⁵ |
| Weinstock (direct) | 45 | 8 | 0.400 | 0.533 | −0.133 | causal p = 0.94 (null) |
| **combined** | 6,167 | 10 | **0.559** | **0.569** | **−0.010** | p = 5×10⁻⁵ |

*(the vs-null p is edge-level; the honest, cluster-robust corroboration is the regulator-level tests below.)*

**Rigor / adversarial checks (`scripts/verify_gf2.py`):**
- **Above-null recovery is real & cluster-robust:** **9/9** regulators have causal acc > 0.5 (binom
  p = 0.004) — TBX21 (n=1 edge, acc 0.0) is excluded, so 9/9 and this p are over 9 of the 10 held-out
  regulators (keeping TBX21 → 9/10, p ≈ 0.02). Regulator cluster-bootstrap causal acc = 0.559,
  **95% CI [0.531, 0.584] excludes 0.5**.
- **No causal-specific advantage — twin is marginally *better*:** only **2/9** regulators have
  causal > twin (binom p = 0.18); cluster-bootstrap **causal−twin = −0.010, 95% CI [−0.013, −0.005]
  excludes 0**; pooled McNemar b=231/c=292, p=0.009. The two models are practically equivalent (56% vs
  57%); the do-operator confers **no** benefit for external edge recovery.
- **Not a convention artifact:** in the two above-chance conventions (derived 0.559/0.569, all-unflipped
  0.561/0.569) twin ≥ causal; all-flipped (0.439/0.431) and swapped (0.441/0.431) are the 1−acc mirror
  images (both below chance), where causal numerically exceeds twin only because the gap reflects. **The
  twin edge holds in every above-chance (meaningful) convention** (derived and unflipped near-identical
  since Freimer dominates).
- **Weinstock direct edges (45) are underpowered** (causal null at p=0.94; twin n.s.) — reported, not
  over-read either way.

**G-F.2 verdict: FAIL (causal-specificity).** The do-operator recovers held-out external edge *directions*
above chance, but this is a property of the **predictive model, not the do-operation** — the non-causal
twin matches (slightly beats) it. This is a genuine, non-circular, well-powered negative.

## Honest framing
- **Positive that IS real:** zero-shot, cross-dataset directional recovery of external edge directions at
  ~56%, robust across all 9 Freimer regulators. **The signal is carried by Freimer's *indirect* KO-DE edges
  (6,122); the 45 *direct*-causal Weinstock edges recover only at chance (0.400, p=0.94, underpowered)** —
  so this is recovery of DE (indirect regulator→target) structure, not of the direct causal edges. The
  model generalizes to unseen regulators' downstream signs.
- **The claim that FAILS:** that this recovery is *causal* (do-operator-specific). It is not — it is
  correlational (twin-equivalent). The within-dataset C2 causal advantage does not transfer externally.
- **Same-lab caveat still applies:** cross-platform/modality, not independent-lab replication.

## What is untouched
Frozen tag `submission-fallback-v1` / `6476670` and CP2 `benchmark_table.csv` are byte-untouched (verified:
tag → 6476670, 0 changes to benchmark_table.csv). The re-train pinned the committed HVG panel + frozen
split (never re-froze). Nothing merged to `main`; C-FUSE 1b lives on `fusion-gates`.

## Reproduce
- `scripts/rebuild_pseudobulk_frozen.py` — restore pseudobulk pinned to the frozen HVG/split (idempotent).
- `scripts/build_priors.py` — ESM2-650M gene-token priors on the frozen panel.
- `scripts/fusion_retrain.py 40` — hold out the 10 regulators, re-train causal+twin, predict zero-shot.
- `scripts/fusion_gf2.py` — powered sign-recovery vs twin vs degree-preserving null.
- `scripts/verify_gf2.py` — adversarial checks (convention, clustering, cluster bootstrap, McNemar).
- `run_fusion_pipeline.sh` — AFK-safe orchestrator (checkpointed; tmux; `FUSION_STATUS.txt` sentinel).

Deliverables: `results/fusion_gf2.csv`, `results/fusion_gf2_perreg.csv`,
`results/fusion_pred_{causal,noncausal}.parquet`, `results/fusion_measurable_edges.csv`,
`results/fusion_gf1_overlap.csv`, `data/external/weinstock_edges.csv`, `data/external/PROVENANCE.md`.
