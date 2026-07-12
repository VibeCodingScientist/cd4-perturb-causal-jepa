# The Predictability Audit — interactive explorer

An interactive, three-act walkthrough of the **v2 predictability audit** — wired to the
**committed result CSVs on `main`**. Every number on screen is read from a CSV; nothing is
hand-typed or synthetic.

## Launch it (one step, no build)

**Simplest — open the self-contained file (zero setup, works offline):**

```
open explorer/explorer_bundle.html          # macOS
# or just double-click explorer/explorer_bundle.html
```

`explorer_bundle.html` inlines everything (JS, fonts, the hero scorecard SVG, and all data)
into a single file that runs from `file://` with no server and no network. It is also a
release asset.

**Or serve the folder** (identical content, if you prefer a dev server):

```
cd explorer && python3 -m http.server 8000   # then open http://localhost:8000/explorer.html
```

That's it — no `npm install`, no build, no API keys.

## What it shows (the audit arc)

| Act | Panel | The point |
|---|---|---|
| **1** | The anchor | **C2** — the do-operator beats its non-causal twin (+0.118 condition / +0.162 gene). It is the audit's **signal-detection positive control**: the null machinery *can* detect signal, so the nulls that follow mean "no signal," not "blunt instrument." Includes the data-integrity control (restored data reproduces C2 within tolerance). |
| **2** | The reframe | Raw Pearson-δ is uninterpretable alone; re-read as **fraction-of-ceiling** the axes dissociate — condition linear-dominated, gene structure-dominated (bucket **C ≈ 0.76**, real at p<0.001). Raw δ is shown *beside* fraction-of-ceiling. |
| **3** | The scorecard | The hero: the committed `predictability_scorecard.svg`, then **seven pre-registered probes** (six at the floor, P7 in-distribution) + the **C2 positive-control anchor**, each with its verdict + key number from its gate CSV, the residual's identity (activation-cytokine program), the honest **Tier-2** novelty frame, and a subordinate **second-dataset port appendix** (Schmidt 2022 — machinery ports, four bounds verbatim). Zero GPU. |

Use the **Eval axis** toggle (condition / gene hold-out) — Acts 1–2 respond to it; Act 3 is
axis-agnostic. The **Next act ›** button walks the arc.

## Provenance — which CSV feeds which panel

Rebuild the data from `main` at any time (needs the repo venv with pandas):

```
<repo>/.venv/bin/python explorer/export_app_json.py
```

It prints a verification table of every headline number and writes `data/*.json`
(all `_meta.source == "real"`, which is why the "demo data" badge is gone).

| Panel | CSV(s) |
|---|---|
| Act 1 · C2 anchor + leaderboard | `results/benchmark_table.csv` |
| Act 1 · data-integrity control | `results/c2_control.csv` |
| Act 1 · localization | `results/do_operator_localization.csv` |
| Act 2 · fraction-of-ceiling | `results/fraction_of_ceiling.csv` |
| Act 2 · A/B/C budget | `results/budget_decomposition.csv` (B = 1−r_ceiling; A = Ridge-frac × r_ceiling; C = r_ceiling − A) |
| Act 2 · bucket C is real | `results/budget_cross_donor.csv` |
| Act 3 · the seven-probe scorecard | `results/predictability_audit_gate.csv` (canonical; each row self-checks against the committed verdict) |
| Act 3 · P7 external causal-edge | `results/fusion_gf2.csv` |
| Act 3 · residual identity | `results/phaseB_top_residual_genes.csv` |
| Act 3 · hero figure | `figures/predictability_scorecard.svg` (embedded verbatim) |
| Act 3 · Schmidt port appendix | `results/gpa2_scorecard.csv` + four bounds verbatim from `GPA2_PORT.md` |

The map is also embedded in `data/manifest.json` (`provenance`). The seven-probe scorecard
resolves to the underlying gate CSVs via `predictability_audit_gate.csv`'s `source` column
(`c4_auroc.csv`, `cnl_realdata_gate.csv`, `phaseB_snr_precheck_summary.csv`,
`trajectory_coupling_gate.csv`, `donor_structure_gate.csv`, `relational_gate.csv`,
`fusion_gf2.csv`).

**Schmidt second-dataset appendix (G-PA.2):** PR #13 is folded into `main`, so Act 3 includes a
**clearly-subordinate** port element — the audit *machinery* ports to Schmidt 2022 (R1 cross-well,
R2 ceiling, R3 relational reproduce above Schmidt's own null, no retrain), with the four bounds
stated **verbatim** from `GPA2_PORT.md` (decisively: cross-well ≠ cross-donor — the floor *finding*
was not re-tested; same lab; CRISPRa vs CRISPRi; 3/7 probes). The v2 **headline is unchanged**.

## Files

```
explorer/
  explorer.html           # entry (dev server)
  explorer_bundle.html     # self-contained offline build (open this) — release asset
  app.js                   # state + router + chart kit (vanilla JS, D3 the only dep)
  style.css                # design tokens + components
  panels/act1-dooperator.js  act2-budget.js  act3-scorecard.js
  export_app_json.py       # committed v2 CSVs -> data/*.json  (source:"real")
  build_single_file.py     # inline everything -> explorer_bundle.html
  data/*.json              # committed, real, verified
  assets/                  # vendored D3 + fonts (offline)
```

## A note on honesty

The badge stays until **every** panel reads real committed values; it is gone here because
they all do. The arc leads with the confirmed do-operator as the positive control, reframes
raw scores against the reliability ceiling, and makes the seven-probe scorecard the
centerpiece — *"we measured, honestly, how predictable this dataset is,"* not *"we solved
perturbation prediction."* Reported as **Tier-2, n=1** (a case study), with the do-operator
edge shown to be in-distribution (P7), not causal.
