# 3-minute demo storyboard — v2, The Predictability Audit

Driven entirely by the explorer (open `explorer_bundle.html`). Two controls: the **Eval
axis** toggle (condition / gene) and **Next act ›**. Every number on screen is read from a
committed v2 CSV. The spine is the audit; the negatives are the story — told straight.

---

**0:00–0:30 · the anchor (Act 1, Gene axis)**
- Point to the **+0.162** C2 hero and the green **Data-integrity control passed** panel.
> "Genome-scale CRISPRi in CD4⁺ T cells. One thing works: the do-operator — a knockdown
> treated as an *intervention*, not an observation — beats its non-causal twin by +0.118 on
> the unseen state and +0.162 on unseen genes, where the linear baseline collapses to 0.02.
> That's the *positive control*: it proves the audit's null machinery can detect real signal.
> So when the rest comes back null, it means 'no signal' — not 'blunt instrument.'"

**0:30–1:15 · the reframe (Next act ›)**
- Show the budget bar on **Condition** (mostly linear). Toggle **Eval axis → Gene**: the bar
  becomes almost all amber (structured 0.76, linear collapsed to 0.01).
> "A raw δ score is uninterpretable on its own — you have to know how much signal was reliably
> *there*. Calibrated to the measured reliability ceiling, the axes dissociate: the condition
> shift is linear; the gene shift is almost entirely *structured* signal — real, reproducing
> across donors above a shuffled null. The do-operator recovers ~56% of it. The rest is a
> located gap."

**1:15–2:30 · the scorecard (Next act ›, the centerpiece)**
- Rest on the committed **scorecard figure**. Then pan the seven interactive probe cards;
  click P1, P6, P7 to expand. End on the green **C2** anchor card.
> "Here is the whole dataset audited on one card. Seven pre-registered probes, each scored
> against its own null and calibrated to the ceiling — an explicit causal matrix, third-moment
> fluctuations, single-cell depth, trajectory geometry, donor structure, relational similarity,
> and external causal edges. Six sit at the floor; the seventh recovers external edges but with
> no advantage over its twin — in-distribution, not causal. And the one accuracy positive, C2,
> is the anchor that makes every null credible. The residual isn't noise — it's the transient
> activation-cytokine program. Mapped seven ways, zero GPU."

**2:30–3:00 · the takeaway**
> "This is the first reliability-ceiling-calibrated, positive-control-anchored predictability
> audit of a Perturb-seq dataset — a diagnostic that tells a biologist what's recoverable
> *before* they burn GPU. It's a methods contribution on the axis the field has explicitly
> asked for, reported honestly as Tier-2, n=1. Not 'we solved perturbation prediction' — 'we
> measured, honestly, how predictable this dataset actually is.'"

---

### Notes for recording
- Keep the tab **focused** (count-up / draw-in animations gate on `document.hidden`).
- Badge is gone — the header shows **✓ real results**. If it ever shows "demo data," re-run
  `export_app_json.py` against `main` before recording.
- The **scorecard figure** is the committed `figures/predictability_scorecard.svg`, embedded
  verbatim — it is the hero; let it sit on screen for a beat.
- Fallback if pressed: Acts 1 → 3 (skip Act 2's toggle) still lands the arc.
