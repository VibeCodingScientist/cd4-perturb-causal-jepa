# CD4+ Perturbation Explorer

An interactive, three-act walkthrough of this repository's results — wired to the
**committed result CSVs on `main`**. Every number on screen is read from a CSV; nothing
is hand-typed or synthetic.

## Launch it (one step, no build)

**Simplest — open the self-contained file (zero setup, works offline):**

```
open explorer/explorer_bundle.html          # macOS
# or just double-click explorer/explorer_bundle.html
```

`explorer_bundle.html` inlines everything (JS, fonts, and all data) into a single file
that runs from `file://` with no server and no network.

**Or serve the folder** (identical content, if you prefer a dev server):

```
cd explorer && python3 -m http.server 8000   # then open http://localhost:8000/explorer.html
```

That's it — no `npm install`, no build, no API keys.

## What it shows (the three-act arc)

| Act | Panel | The point |
|---|---|---|
| **1** | The do-operator works | **C2** — the causal do-mask beats its non-causal twin (+0.118 condition / +0.162 gene), and the edge concentrates on *reliable* perturbations. Intervention, not observation. |
| **2** | The predictability budget | Raw Pearson-δ is baseline-dominated; re-read as **fraction-of-ceiling** the axes dissociate — condition is linear-dominated, gene is structure-dominated (bucket **C ≈ 0.76**, real at p<0.001). Raw δ is shown *beside* fraction-of-ceiling. |
| **3** | The frontier, six ways | The ~0.03 pointwise floor, triangulated by **six pre-registered negatives** (each with its verdict + key number from its gate CSV), and the residual's biological identity — the transient activation-cytokine program. Zero GPU. |

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
| Act 1 · C2 + leaderboard | `results/benchmark_table.csv` |
| Act 1 · localization | `results/do_operator_localization.csv` |
| Act 2 · fraction-of-ceiling | `results/fraction_of_ceiling.csv` |
| Act 2 · A/B/C budget | `results/budget_decomposition.csv` (B = 1−r_ceiling; A = Ridge-frac × r_ceiling; C = r_ceiling − A) |
| Act 2 · bucket C is real | `results/budget_cross_donor.csv` |
| Act 3 · residual identity | `results/phaseB_top_residual_genes.csv` |
| Act 3 · floor + single-cell gate | `results/phaseB_snr_precheck_summary.csv` |
| Act 3 · gate 1 causal-matrix | `mechanism/results/c4_auroc.csv` |
| Act 3 · gate 2 fluctuation | `mechanism/results/cnl_realdata_gate.csv` |
| Act 3 · gate 4 trajectory | `results/trajectory_coupling_gate.csv` |
| Act 3 · gate 5 donor-structure | `results/donor_structure_gate.csv` |
| Act 3 · gate 6 relational | `results/relational_gate.csv` |

The map is also embedded in `data/manifest.json` (`provenance`).

## Files

```
explorer/
  explorer.html          # entry (dev server)
  explorer_bundle.html    # self-contained offline build (open this)
  app.js                  # state + router + chart kit (vanilla JS, D3 the only dep)
  style.css               # design tokens + components
  panels/act1-dooperator.js  act2-budget.js  act3-frontier.js
  export_app_json.py      # committed CSVs -> data/*.json  (source:"real")
  build_single_file.py    # inline everything -> explorer_bundle.html
  data/*.json             # committed, real, verified
  assets/                 # vendored D3 + fonts (offline)
```

## A note on honesty

The badge stays until **every** panel reads real committed values; it is gone here because
they all do. The arc leads with the confirmed do-operator, reframes raw scores as
fraction-of-ceiling, and makes the six-negative frontier map the centerpiece — *"we bounded
this honestly, six ways,"* not *"we solved perturbation prediction."*
