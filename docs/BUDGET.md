# The Predictability Budget of the CD4⁺ CRISPRi Benchmark

*Developer 2 · C-BUDGET · CPU-only analysis of committed artifacts (no retraining, CP2 untouched)*

**Summary.** A perturbation-prediction score such as "Pearson-δ 0.42" is uninterpretable in isolation, because it does not convey how much of the response is *predictable in principle*. This analysis partitions each perturbation's top-50-DEG response into three exhaustive, disjoint buckets on the evaluation metric's own (Pearson-δ) scale — **B**, irreducible replicate noise; **A**, linear-explainable signal; and **C**, structured-but-unmodeled signal — and asks where the models and the do-operator actually reside. The principal findings are that the task is not noise-saturated (the noise floor is only 22–33%), and that its difficulty lies almost entirely on the gene-generalization axis. There, the achievable signal is *structured* (linear models collapse) and the do-operator is the only model that recovers it, yet even the do-operator leaves approximately 44% of the achievable gene signal uncaptured. That residual constitutes a located open target rather than a dead end.

---

## 1 · Method

For each evaluable perturbation, the achievable Pearson-δ over its own top-50 true DEGs (the frozen evaluation metric) is decomposed into three pieces that sum to 1:

| Bucket | Definition | How measured | Provenance |
|---|---|---|---|
| **B** — irreducible noise floor | `1 − r_ceiling` | split-half reliability over (guide × donor) replicate units, Spearman-Brown-disattenuated, in the frozen normalization/control space | **[IN-PROJECT]** Stage 1 |
| **A** — linear-explainable | `frac_of_ceiling(Ridge) × r_ceiling` | Ridge is a fitted linear map; its held-out fraction-of-ceiling is a *predictive* (not in-sample) estimate of the linear-reachable signal — an upper bound on linear | **[IN-PROJECT]** Stages 1–2 |
| **C** — structured residual | `r_ceiling − A` | what remains inside the reliability ceiling that *no linear model* reaches | **[INFERENCE]** |

`r_ceiling` is the maximum Pearson-δ that any model could score against the noisy held-out ground truth. B, A, and C constitute a decomposition of the evaluation metric's achievable correlation, a monotone predictability proxy, rather than an orthogonal variance partition. The analysis deliberately does not set A equal to CIPHER's in-sample R² (in accordance with the brief's warning against a naive `1−R²`); CIPHER Σu (Developer 4's committed mechanistic-linear R² ≈ 0.30) is shown only as a reference tick.

---

## 2 · Bucket B — the noise floor is modest, and the same in both regimes

Split-half reliability (n = 20 random halves per perturbation, Spearman-Brown to full sample):

| Split | median `r_ceiling` (achievable) | **B = noise floor** | specific-residual reliability | n |
|---|---|---|---|---|
| Condition hold-out (Stim48hr) | 0.669 | **0.331** | 0.667 | 2 184 |
| Gene hold-out (unseen targets) | 0.776 | **0.224** | 0.775 | 315 |

The response is highly reproducible. Only 22–33% of the top-50 signal is lost to replicate noise, and, critically, the *perturbation-specific* residual (after subtracting the shared activation program) is nearly as reliable as the full response (0.67 / 0.78). Consequently, most of the achievable ceiling is genuine, and most of it is perturbation-specific rather than the shared program. A ceiling far from saturation therefore exists to be climbed.

## 3 · Bucket C is real — cross-donor permutation null

The question is whether the perturbation-specific residual (the substrate of C) is reproducible biology or a within-donor/batch artifact. For each perturbation, the donor-mean *specific* residual (δ − shared program) was taken and its cross-donor correlation measured against a null that shuffles which perturbation's residual sits under each donor label (1 000 permutations).

| Split | real cross-donor specific-r | null mean | null p95 | empirical p |
|---|---|---|---|---|
| Condition | **0.0317** | 0.0005 | 0.0018 | **< 0.001** |
| Gene | **0.0492** | −0.0005 | 0.0029 | **< 0.001** |

The specific residual reproduces across donors approximately 60–100× above the shuffled null (p < 0.001 in both splits). The absolute value is small because it is a full 3000-HVG cross-donor correlation of the residual after removing the dominant shared program — that is, a significance test rather than the bucket-C magnitude, which lives on the top-50 ceiling scale in §4. C therefore represents donor-invariant reproducible structure, not batch.

## 4 · Bucket A versus C is split-dependent — the reframing

Fraction-of-ceiling = model's median Pearson-δ ÷ its achievable ceiling (cluster-bootstrap 95% CI):

| Model | Condition raw δ | Condition **×ceiling** | Gene raw δ | Gene **×ceiling** |
|---|---|---|---|---|
| Ridge (linear) | 0.427 | **0.67** | 0.013 | **0.02** |
| do-operator (causal) | 0.352 | 0.55 | 0.420 | **0.56** |
| JEPA→causal | 0.354 | 0.55 | 0.419 | 0.55 |
| JEPA-only | 0.254 | 0.39 | 0.301 | 0.39 |
| non-causal | 0.240 | 0.37 | 0.226 | 0.30 |
| FCN | 0.091 | 0.14 | 0.132 | 0.16 |

Feeding Ridge's fraction-of-ceiling into the bucket split yields opposite budgets for the two generalization axes:

| | **A** (linear) | **C** (structured) | **B** (noise) |
|---|---|---|---|
| **Condition** hold-out | **≈ 0.45** | ≈ 0.22 | ≈ 0.33 |
| **Gene** hold-out | **≈ 0.01** | **≈ 0.76** | ≈ 0.22 |

- **Condition shift** (zero-shot to a new timepoint): a *linear* response model already reaches two-thirds of the ceiling. The reproducible signal is mostly linear-explainable, C is a modest slice, and the headroom is limited.
- **Gene shift** (unseen perturbation targets): Ridge collapses to the noise floor (0.02 of ceiling). Almost the entire achievable signal is structured (C ≈ 0.76) and is *not* linear-transferable from seen genes. This is where the benchmark is genuinely hard.

Two models with near-identical raw δ (Ridge 0.427 on condition, causal 0.420 on gene) achieve very different fractions of what is achievable, by entirely different mechanisms — precisely what raw Pearson-δ conceals and fraction-of-ceiling exposes.

## 5 · Where does the do-operator reside? (localizing C2)

Per-perturbation do-operator edge `C2 = causal − non-causal`, split by reliability:

| Split | C2 median | C2 on **reliable** perts | C2 on unreliable perts | corr(C2, r_ceiling) | causal − Ridge |
|---|---|---|---|---|---|
| Condition | +0.026 | +0.015 | +0.034 | +0.10 | **−0.069** |
| Gene | +0.041 | **+0.141** | +0.022 | **+0.29** | **+0.370** |

- **Gene axis:** the do-operator's advantage concentrates on the *reliable* perturbations (+0.141 versus +0.022; corr +0.29) and exceeds Ridge by +0.37. Concentration on high-reliability perturbations is the signature of a model recovering real reproducible bucket-C structure rather than fitting noise. On the ceiling scale it recovers ≈ 0.43 of 0.78 → approximately 56% of the achievable gene signal (fraction of the reliability ceiling); measured against the structured bucket-C residual that no linear model touches, it recovers approximately 54% (0.544, [`phaseB_recovery.csv`](../results/phaseB_recovery.csv)).
- **Condition axis:** the do-operator adds little (+0.026) and *loses* to Ridge (−0.069); when the signal is linear, the do-mask confers no benefit. Its value is specifically the **cross-gene structured** regime.

## 6 · Pre-registered go/no-go read

> Registered branches: **(1)** reproducible C small versus B → predictability-ceiling result;
> **(2)** C large and unrecovered → located open target; **(3)** C large but the do-operator
> already recovers it → C2 captures far-from-equilibrium structure.

**Verdict — branch (1) rejected; a split-dependent (2)+(3) blend.**

- The benchmark is not noise-saturated: B is only 0.22–0.33 and no model saturates the ceiling. Branch (1) is rejected.
- On the **gene axis**, C is large (≈ 0.76), *real* (§3, p < 0.001), and the do-operator recovers a reliability-concentrated portion of it (§5) → **branch (3)** for the recovered part (C2 = genuine cross-gene structured signal), and **branch (2)** for the remainder: approximately 44% of the achievable gene signal is reproducible, structured, and unmodeled by every model considered here — a concrete, located open target.
- On the **condition axis**, the reproducible signal is mostly linear (A ≈ 0.45 of the 0.67 ceiling); the modest C (≈ 0.22) is currently unrecovered by *any* model, the do-operator included.

**Benchmark implication.** Scores should be reported as fraction-of-ceiling, per axis. The scientific frontier is the **gene-generalization structured residual**; the do-operator is the current front-runner there and the only model whose edge localizes to reproducible signal, but the ceiling demonstrates that roughly as much structured signal *again* remains to be captured.

## 7 · Figure

`figures/budget_ceiling.png` — stacked A|C|B budget per context on the Pearson-δ scale, with the do-operator's fraction-of-ceiling and the CIPHER Σu reference located. The condition bar is teal-heavy (linear); the gene bar is almost entirely amber (structured), with the do-operator line sitting deep inside bucket C.

## 8 · Provenance and what this analysis does *not* claim

- **[IN-PROJECT]** noise floor, reliabilities, permutation null, fraction-of-ceiling, localization, and bucket split were all computed here from the committed CZI pseudobulk, the frozen split, and the committed `runs/`. Files: `results/budget_reliability.csv`, `budget_cross_donor.csv`, `fraction_of_ceiling.csv`, `do_operator_localization.csv`, `budget_decomposition.csv`.
- **[IN-PROJECT, consumed]** CIPHER Σu R² ≈ 0.30 is Developer 4's committed aggregate, used only as a reference tick (not to define A). A per-split CIPHER would refine the A/C boundary.
- **[INFERENCE]** the A/C split treats Ridge's held-out fraction-of-ceiling as the linear-reachable share; C is the complement inside the reliability ceiling.
- **Not claimed:** reliability ≠ predictability (r_ceiling is an *upper bound* on any model's achievable δ, not a guarantee that one exists); no new method, architecture, loss, or split; no non-equilibrium model (Part IV is explicitly out of scope). Bucket sizes are on the evaluation metric's correlation scale, not an orthogonal variance partition. CP2's committed numbers are untouched.

*Reproduce:* `python scripts/budget_reliability.py && python scripts/budget_stage2.py && python scripts/budget_stage3.py` (CPU, ~15 min, dominated by two CZI reads for the permutation null).
