# 3-minute demo storyboard — the three-act arc

Driven entirely by the explorer (open `explorer_bundle.html`). Two controls: the **Eval
axis** toggle (condition / gene) and **Next act ›**. Every number on screen is read from a
committed CSV. Narration is AI-first / passive; the negatives are the story — told straight.

---

**0:00–0:20 · Cold open (Act 1 on screen, Condition axis)**
> "This is genome-scale CRISPRi in primary human CD4⁺ T cells — 22 million cells, every
> gene silenced one at a time. The question isn't just 'can we predict a knockdown's
> effect' — it's *where the limit of predictability actually is.* Three acts."

**0:20–1:00 · Act 1 — the do-operator works (the positive, first)**
- Point to the **+0.118** hero. Toggle **Eval axis → Gene**: hero flips to **+0.162**; the
  zero-shot stats appear (causal **0.368**, Ridge **0.019**).
> "Same architecture, causal mask on versus off — the do-mask masks only the edges *into*
> the perturbed gene, so the intervention propagates downstream. That's a +52% gain on the
> unseen activation state, +79% on unseen genes, where the linear baseline collapses. It's
> the only model whose edge concentrates on the *reliable* perturbations — signal, not noise."

**1:00–1:50 · Act 2 — the predictability budget (Next act ›)**
- Show the budget bar on **Condition** (mostly teal — linear). Toggle **Eval axis → Gene**:
  the bar becomes almost all amber — **C structured 0.76**, linear collapsed to 0.01.
- Point to the raw-δ-beside-fraction-of-ceiling table, then the **cross-donor 0.049 vs
  null 0.003** stat.
> "Raw scores hide this. Read as fraction-of-ceiling, the two axes dissociate: the condition
> shift is linear-dominated; the gene shift is almost entirely *structured* signal — real,
> reproducing across donors ~17× above a shuffled null, p<0.001. The do-operator recovers
> ~56% of it. The other ~44% is a located gap — that's the frontier."

**1:50–2:50 · Act 3 — the frontier, mapped six ways (Next act ›, the centerpiece)**
- Rest on the **0.033** floor. Pan the six-gate grid; click two or three cards to expand
  (causal-matrix **FAIL**, fluctuation **≈0.000**, relational **0.008**). Then the
  activation-cytokine gene chips.
> "The remaining per-perturbation signal sits at a ~0.03 floor. Six *pre-registered*
> analyses each tried to push past it — an explicit causal matrix, third-moment fluctuations,
> single-cell resolution, trajectory geometry, donor structure, relational similarity —
> spanning pointwise and relational, raw and specific. All six return clean negatives, and
> they converge on the same floor. And the residual isn't noise — it's the transient
> activation-cytokine program: IFNG, IL2, CSF2, the chemokines."

**2:50–3:00 · Close**
> "Six negatives, one floor, and its biological identity — reached with zero GPU. We bounded
> the frontier honestly. We did not solve perturbation prediction — and the map shows exactly
> why that's the harder, truer result."

---

### Notes for recording
- Keep the tab **focused** (the count-up / draw-in animations gate on `document.hidden`).
- The demo badge is gone — the header shows **✓ real results**. If it ever shows "demo
  data," re-run `export_app_json.py` against `main` before recording.
- Fallback if pressed for time: Acts 1 → 3 (skip Act 2's toggle), still lands the arc.
