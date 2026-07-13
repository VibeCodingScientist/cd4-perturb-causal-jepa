# 3-minute demo storyboard — v2, The Predictability Audit

This walkthrough is driven entirely by the explorer (open `explorer_bundle.html`). It exposes two controls: the **Eval
axis** toggle (condition / gene) and **Next act ›**. Every number displayed on screen is read from a
committed v2 CSV. The spine of the presentation is the audit; the negative results constitute the narrative, and they are reported without embellishment.

---

**0:00–0:30 · the anchor (Act 1, Gene axis)**
- Direct attention to the **+0.162** C2 result and the green **Data-integrity control passed** panel.
> "This is genome-scale CRISPRi in CD4⁺ T cells. One component succeeds: the do-operator — a knockdown
> treated as an *intervention* rather than an observation — exceeds its non-causal counterpart by +0.118 on
> the unseen state and +0.162 on unseen genes, where the linear baseline collapses to 0.02.
> This serves as the *positive control*, demonstrating that the audit's null machinery is able to detect genuine signal.
> Consequently, when the remaining probes return null, the result indicates an absence of signal rather than an insensitive instrument."

**0:30–1:15 · the reframe (Next act ›)**
- Display the budget bar on **Condition** (predominantly linear). Toggle **Eval axis → Gene**: the bar
  becomes almost entirely amber (structured 0.76, linear collapsed to 0.01).
> "A raw δ score is not interpretable in isolation; one must first establish how much signal was reliably
> present. Once calibrated to the measured reliability ceiling, the two axes dissociate: the condition
> shift is linear, whereas the gene shift is almost entirely *structured* signal — genuine, and reproducing
> across donors above a shuffled null. The do-operator recovers approximately 56% of it, and the remainder
> is a located gap."

**1:15–2:30 · the scorecard (Next act ›, the centerpiece)**
- Rest on the committed **scorecard figure**. Then pan across the seven interactive probe cards, and
  click P1, P6, and P7 to expand them. Conclude on the green **C2** anchor card.
> "This is the entire dataset audited on a single card. Seven pre-registered probes, each scored
> against its own null and calibrated to the ceiling: an explicit causal matrix, third-moment
> fluctuations, single-cell depth, trajectory geometry, donor structure, relational similarity,
> and external causal edges. Six sit at the floor; the seventh recovers external edges but with
> no advantage over its twin, and is therefore in-distribution, not causal. The single accuracy positive, C2,
> is the anchor that renders every null result credible. The residual is not noise; it is the transient
> activation-cytokine program, mapped seven ways, with zero GPU."
- *(Optional, scroll to the subordinate appendix)* The audit **machinery** ports to a second
  dataset (Schmidt 2022) — R1/R2/R3 reproduce above its own null, without retraining — but this material should be kept
  subordinate, and **BOUND 1** should be read aloud: cross-well ≠ cross-donor, so the *floor finding* was
  not re-tested. The headline is unchanged.

**2:30–3:00 · the takeaway**
> "This is the first reliability-ceiling-calibrated, positive-control-anchored predictability
> audit of a Perturb-seq dataset: a diagnostic that informs a biologist what is recoverable
> *before* GPU resources are expended. It is a methods contribution on the axis that the field has explicitly
> requested, reported honestly as Tier-2, n=1. The claim is not that perturbation prediction has been solved, but rather
> that the predictability of this particular dataset has been measured honestly."

---

### Notes for recording
- Keep the tab **focused** (count-up and draw-in animations gate on `document.hidden`).
- The badge has been removed; the header now shows **✓ real results**. If it ever shows "demo data," re-run
  `export_app_json.py` against `main` before recording.
- The **scorecard figure** is the committed `figures/predictability_scorecard.svg`, embedded
  verbatim. It is the centerpiece, so allow it to remain on screen for a moment.
- Fallback if time is constrained: Acts 1 → 3 (skipping Act 2's toggle) still conveys the arc.
