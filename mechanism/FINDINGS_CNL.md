# C-NL Gate — does the baseline third moment predict what covariance cannot?

**Verdict: LIVE** (all three tests pass). The first positive result in this line — hand off to the
project lead for the go/no-go on a real-CD4 build.

## 0. Provenance discipline (a rule, not a preamble)

The motivating physics claim is an **inference**, not a cited result, and is treated as such throughout:

- **Verified (CIPHER):** CIPHER's model is exactly first-order, `ΔX = Σu` (response = baseline covariance ×
  perturbation input), and its linear-response error grows with perturbation magnitude and diverges past a
  critical coupling.
- **NOT in CIPHER:** the words "third moment / cumulant / skew / susceptibility" do not appear; CIPHER never
  attributes its breakdown to a third-moment correction.
- **The inference (tested here, not cited):** standard second-order fluctuation-response theory identifies the
  leading correction to `Σu` as a contraction of the baseline third cumulant with the perturbation. This gate
  converts that inference into evidence on ground truth. Nowhere is it claimed that "CIPHER shows the third
  moment…"; `T[u,u]` is a **candidate predictor** whose coefficient is **fit**, never assumed to be the
  analytic ½.

## 1. Construction (equilibrium regime, genuine non-Gaussianity)

- **Symmetric `A`** (`make_A_symmetric`): the linear stationary covariance is `Σ = −(σ²/2)A⁻¹`, so **σ²=2**
  makes CIPHER's first-order response `Σu` equal the true response `Δμ = −A⁻¹u` **exactly at λ=0** — any λ>0
  residual is then genuinely nonlinear.
- **Genuine non-Gaussian sampling:** the third moment needs the true non-Gaussian stationary law, so we
  integrate the nonlinear SDE `dx = (A h_λ(x) + b + Γ)dt + σ dW` by Euler–Maruyama (`s=0.4`). The spike-2
  Gaussian local-covariance sampler has zero third moment by construction and is unusable here.
- **Nonzero baseline `b~N(0,1)` (construction lesson, documented):** with `b=0` the operating point sits at
  `x*=0`; because `tanh` is **odd**, the stationary law around 0 is symmetric → third moment ≈ 0 by symmetry
  and the quadratic response vanishes. A first (buggy) run showed exactly this (`‖c‖≈0` at all λ). Moving the
  operating point off 0 with `b≠0` restores both. This does **not** affect M0 (λ=0 is linear regardless of `b`).

## 2. What the tests measure

- **Pure second-order response** (Σ-independent; cancels the linear part and any χ≠Σ non-equilibrium mismatch
  exactly): `c_ik = [Δμ_i(+m e_k) + Δμ_i(−m e_k)] / (2 m²)`, via common-random-number CRN pairs. `c_ik = 0` at
  λ=0 by construction. (This is a cleaner estimator than the raw `Δμ − Σu` residual, whose λ=0 value is
  dominated by covariance *sampling* noise — see the M0 calibration.)
- **Test 1:** does `‖c‖` and `‖T‖` rise with λ?
- **Test 2:** does the third-moment feature `T[e_k,e_k]_i = T_ikk` predict `c_ik` better than the covariance
  surrogate `(Σe_k)_i² = Σ_ik²`? Pass = `ΔR² > 0`, 8-seed cluster-bootstrap CI excluding 0 (both single fitted
  coefficient — equal capacity).
- **Test 3:** re-estimate `T̂` at reduced depth — latent subsample (pure third-moment variance) and with the
  NB **emission on** (realistic observation) — and find where `ΔR²` stops excluding 0.

## 3. Results

### Tests 1 + 2 — ΔR² vs λ (8 seeds; `results/delta_r2_vs_lambda.{csv,png}`)

| λ | ‖c‖ (2nd-order resp.) | ‖T‖ (3rd moment) | R²_T | R²_cov | **ΔR²** [95% CI] |
|---|---|---|---|---|---|
| 0.00 | ~0 | 0.0003 | 0.000 | 0.000 | **−0.000** [−0.002, +0.000] |
| 0.25 | 0.0003 | 0.0004 | 0.614 | 0.000 | **+0.614** [+0.547, +0.663] ✓ |
| 0.50 | 0.0011 | 0.0009 | 0.734 | 0.000 | **+0.733** [+0.673, +0.772] ✓ |
| 0.70 | 0.0031 | 0.0024 | 0.756 | 0.002 | **+0.754** [+0.684, +0.793] ✓ |
| 0.85 | 0.0097 | 0.0075 | 0.756 | 0.008 | **+0.749** [+0.640, +0.802] ✓ |

**M0 clean** (λ=0: `‖c‖≈0`, ΔR²≈0). **Test 1 passes** (`‖c‖`, `‖T‖` both rise with λ). **Test 2 passes
decisively**: the baseline third moment explains ~61–76% of the second-order response; the covariance
surrogate explains ~0%. ΔR² CI excludes 0 at every λ>0.

### Test 3 — estimability vs depth at λ=0.85 (`results/depth_threshold.{csv,png}`)

| control cells | ΔR² latent (pure variance) [CI] | ΔR² emission-on (realistic) [CI] |
|---|---|---|
| 100,000 | +0.749 [+0.632, +0.800] | +0.250 [+0.128, +0.322] |
| 30,000 | +0.745 [+0.631, +0.792] | +0.246 [+0.120, +0.320] |
| 10,000 | +0.728 [+0.616, +0.777] | +0.241 [+0.120, +0.316] |
| 3,000 | +0.695 [+0.587, +0.745] | +0.223 [+0.111, +0.287] |
| 1,000 | +0.614 [+0.512, +0.651] | +0.185 [+0.090, +0.234] |

**Test 3 passes**: ΔR² CI excludes 0 at **every** depth down to 1,000 cells, both for the pure-variance
(latent) and the realistic (NB emission on) estimators. No depth floor is reached in the tested range —
the signal is estimable even at low depth. NB emission is not free: it costs ~2/3 of the signal
(ΔR² 0.75 → 0.25 at full depth), but the third moment still clearly beats covariance.

## 4. Readout

All three tests pass → **LIVE**. On symmetric-`A` ground truth, the baseline third moment of unperturbed
fluctuations predicts the second-order perturbation response that the first-order (covariance) model leaves
unexplained — decisively (ΔR² ≈ +0.6–0.75, CI excluding 0) and estimably (survives to 1,000 control cells,
and survives NB emission at a ~2/3 signal cost). This is the one opening the two mechanism-recovery negatives
left: covariance/Lyapunov sufficiency is a *second-moment* property, and the perturbation-response information
it cannot carry lives, as inferred, in the third moment. Two honest qualifications gate the interpretation.
**(1) Magnitude:** the nonlinear term is *small* — the second-order contribution is only ~3–4% of the total
response at λ=0.85 (`‖c‖·m² ≈ 0.06` vs `‖Δμ‖ ≈ 1.7`). The gate tests *prediction* (R²), which is strong;
whether the term is *large enough to matter* on real data is a separate sizing question (CIPHER Fig-2G — the
lead's call, not the gate's). Small-but-structured is a positive result; it is not a claim of large effect.
**(2) Equilibrium:** the gate deliberately runs where the theory is exact (symmetric/gradient `A`); real gene
networks are non-equilibrium, so a LIVE here is **necessary, not sufficient** for real data.

## 5. Go / no-go (the lead's decision, stated per the brief)

LIVE unlocks — **not built here** — the real-CD4 step: closed-form `ΔX = Σ_c u + c·T_c[u,u]`, a low-rank `T`
estimator (CIPHER found responses propagate through ~3 global modes; low-rank both denoises and matches that
structure), and CIPHER + additive baselines on the frozen `core.eval`. Before any real-data claim, the
second-order term's exact analytic form should be derived separately (this gate only fits `c`). Out of scope
by rule: real ZHU25 data, the analytic-½ assumption, the low-rank estimator, anything downstream.

## 6. Risk register carried forward (not the gate's concern)

- **[CRUX] Third-moment estimation variance** — sized here: survives to 1,000 cells on ground truth; low-rank
  `T` and this dataset's large control depth are the real-data mitigations.
- **[MAGNITUDE]** the nonlinear term is small (~3–4% of the response) — real-data size is unconfirmed.
- **[EQUILIBRIUM]** exact only for gradient systems; real networks are non-equilibrium — necessary-not-sufficient.
- **[INTERPRETATION]** "third moment" is a fluctuation-response predictor, **not** causation; it does not
  resurrect `Â_C` and deliberately avoids estimating `A`.

## Milestones

- **M0** construction: symmetric `A`, σ²=2 → λ=0 response exactly linear (`‖c‖≈0`), EM stable ✓.
- **M1** Test 1: `‖c‖`, `‖T‖` rise with λ ✓.
- **M2** Test 2: ΔR²-vs-λ, CI excludes 0 at every λ>0 ✓.
- **M3** Test 3: depth threshold — survives to 1,000 cells latent and emission-on ✓.
- **Done** — this readout.

## Reproduce

```bash
python run_cnl_gate.py           # full gate (8 seeds) -> results/delta_r2_vs_lambda.{csv,png}, depth_threshold.{csv,png}
python run_cnl_gate.py quick     # fast directional read (reduced; not the committed artifact)
```
CPU-only.

## Update — real-data outcome (does NOT transfer)

The go/no-go this gate deferred to the project lead has been run on the real CD4⁺ CRISPRi data
(4 donors × 3 conditions, 16,188 perturbations, CIPHER-exact raw-count `Σ`/`ΔX`). **Result: NEGATIVE
in all 12 strata** — the diagonal third-moment feature is orthogonal to the CIPHER residual
(mean ΔR² +0.0000, jackknife 95% CI [−0.0000, +0.0000]), the opposite of the LIVE result here. The
simulator's signal does not survive real single-cell estimation error. Full writeup:
[FINDINGS_CNL_REALDATA.md](FINDINGS_CNL_REALDATA.md). No-go on a third-moment closed-form for real data.
