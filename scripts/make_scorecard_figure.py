#!/usr/bin/env python
"""Generate figures/predictability_scorecard.svg — Panel A is DATA-DRIVEN from the committed
results/fraction_of_ceiling.csv (gene hold-out, frac_of_ceiling_median), so every bar is on the same unit
(fraction of the measured reliability ceiling) and cannot silently drift to a raw-delta value. Panel B (the
seven-probe verdict scorecard) is a verified static template. Stdlib-only.

Run:  python scripts/make_scorecard_figure.py   ->  figures/predictability_scorecard.svg
"""
import csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X0, SCALE = 300, 600            # x origin; 600 px = fraction 1.0 (ceiling line at x=900)

# display order (top→bottom) with y, bar colour, bold flag
ROWS = [
    ("ridge",       "Ridge (linear)",       192, "#b04a4a", False),
    ("fcn",         "FCN (nonlinear)",       220, "#9aa1ab", False),
    ("noncausal",   "Non-causal twin",       248, "#7e94c4", False),
    ("jepa_only",   "JEPA-only",             276, "#9aa1ab", False),
    ("jepa_causal", "JEPA + causal",         304, "#3f9e6b", False),
    ("causal",      "Do-operator (causal)",  332, "#2e8b57", True),
]


def frac_of_ceiling_gene():
    out = {}
    with open(os.path.join(ROOT, "results", "fraction_of_ceiling.csv")) as f:
        for r in csv.DictReader(f):
            if r["split"] == "gene":
                out[r["model"]] = float(r["frac_of_ceiling_median"])
    return out


def panel_a(frac):
    parts = ['  <g font-size="13">']
    for model, name, y, colour, bold in ROWS:
        v = frac[model]
        w = v * SCALE
        lblcol = colour if (bold or model in ("ridge", "jepa_causal")) else "#6b7280"
        wt = ' font-weight="700"' if bold else ""
        namewt = ' font-weight="700"' if bold else ""
        namecol = "#12151a" if bold else "#3b424c"
        vtext = f"{v:.2f} — collapses" if model == "ridge" else f"{v:.2f}"
        lblwt = ' font-weight="700"' if (bold or model == "ridge") else ""
        parts.append(f'    <!-- {model} {v:.4f} of ceiling (results/fraction_of_ceiling.csv gene) -->')
        parts.append(f'    <text x="290" y="{y+11}" text-anchor="end" fill="{namecol}"{namewt}>{name}</text>')
        parts.append(f'    <rect x="{X0}" y="{y}" width="{w:.1f}" height="16" rx="2" fill="{colour}"/>')
        parts.append(f'    <text x="{X0+w+8:.0f}" y="{y+13}" fill="{lblcol}"{lblwt}>{vtext}</text>')
    parts.append('  </g>')
    return "\n".join(parts)


HEADER = '''<svg viewBox="0 0 1000 760" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
  <rect x="0" y="0" width="1000" height="760" fill="#ffffff"/>
  <text x="40" y="46" font-size="26" font-weight="700" fill="#12151a">Marson CD4 Perturb-seq — Predictability Scorecard</text>
  <text x="40" y="72" font-size="14.5" fill="#4a515b">Seven pre-registered probes + the do-operator control, calibrated to the measured reliability ceiling. An evaluation/methods reframe of validated content — no new model.</text>
  <rect x="40" y="84" width="255" height="22" rx="4" fill="#eef1f5" stroke="#c9d0da"/>
  <text x="49" y="99.5" font-size="12.5" fill="#3b424c">Novelty tier: <tspan font-weight="700">Tier-2</tspan> — a predictability characterization</text>
  <text x="960" y="99.5" font-size="12" fill="#8a8f98" text-anchor="end">reproduced from committed gate CSVs · n=1 dataset (case study)</text>
  <text x="40" y="146" font-size="16" font-weight="700" fill="#12151a">A · Fraction of the measured reliability ceiling recovered — gene hold-out</text>
  <text x="40" y="166" font-size="12.5" fill="#6b7280">Where linear collapses (Ridge 0.02) the do-operator reaches 0.56. The causal−twin gap is the C2 positive.</text>
  <g font-size="12" fill="#8a8f98">
    <line x1="300" y1="186" x2="300" y2="404" stroke="#d5dae1"/>
    <line x1="900" y1="186" x2="900" y2="404" stroke="#c9d0da" stroke-dasharray="4 3"/>
    <text x="300" y="420" text-anchor="middle">0</text>
    <text x="600" y="420" text-anchor="middle">0.5</text>
    <text x="900" y="420" text-anchor="middle">1.0 = ceiling</text>
  </g>'''

FOOTER = '''  <path d="M 484 372 L 484 366 L 633 366 L 633 372" fill="none" stroke="#2e8b57" stroke-width="1.3"/>
  <text x="558" y="390" text-anchor="middle" font-size="12.5" fill="#2e8b57" font-weight="700">C2 do-operator effect = causal − twin (+0.162 gene, Pearson-δ)</text>
  <text x="40" y="470" font-size="16" font-weight="700" fill="#12151a">B · Seven probes — where the recoverable signal is (each vs its own degree/label-preserving null)</text>
  <g font-size="13">
    <text x="40" y="502" fill="#12151a" font-weight="600">P1 · Causal-matrix (Â_C)</text>
    <text x="250" y="502" fill="#5a616b">Â AUROC 0.62 &lt; correlation null 0.83 · mechanism not recovered under P≪G</text>
    <rect x="852" y="490" width="108" height="19" rx="9.5" fill="#f2dede"/><text x="906" y="503.5" text-anchor="middle" fill="#b04a4a" font-weight="700">FAIL</text>
    <text x="40" y="528" fill="#12151a" font-weight="600">P2 · Fluctuation (3rd-moment)</text>
    <text x="250" y="528" fill="#5a616b">ΔR²≈2.8e-6 beyond covariance · 0/24 strata significant</text>
    <rect x="852" y="516" width="108" height="19" rx="9.5" fill="#eceef1"/><text x="906" y="529.5" text-anchor="middle" fill="#6b7280" font-weight="700">NEGATIVE</text>
    <text x="40" y="554" fill="#12151a" font-weight="600">P3 · Single-cell SNR</text>
    <text x="250" y="554" fill="#5a616b">x-donor 0.033 · 16% of cytokine genes SNR&gt;3 · single-cell won't lift it</text>
    <rect x="852" y="542" width="108" height="19" rx="9.5" fill="#eceef1"/><text x="906" y="555.5" text-anchor="middle" fill="#6b7280" font-weight="700">FLOOR</text>
    <text x="40" y="580" fill="#12151a" font-weight="600">P4 · Trajectory-geometry</text>
    <text x="250" y="580" fill="#5a616b">partial ρ 0.007/0.034 (p=0.75/0.55) · not a geometry artifact</text>
    <rect x="852" y="568" width="108" height="19" rx="9.5" fill="#eceef1"/><text x="906" y="581.5" text-anchor="middle" fill="#6b7280" font-weight="700">NEGATIVE</text>
    <text x="40" y="606" fill="#12151a" font-weight="600">P5 · Donor-structure</text>
    <text x="250" y="606" fill="#5a616b">conditioning 0.016 &lt; averaging 0.034 · concordance real but floor-magnitude</text>
    <rect x="852" y="594" width="108" height="19" rx="9.5" fill="#eceef1"/><text x="906" y="607.5" text-anchor="middle" fill="#6b7280" font-weight="700">NO-GO</text>
    <text x="40" y="632" fill="#12151a" font-weight="600">P6 · Relational-object</text>
    <text x="250" y="632" fill="#5a616b">best specific object 0.11 &lt; 0.30 · floor is object-general</text>
    <rect x="852" y="620" width="108" height="19" rx="9.5" fill="#f2dede"/><text x="906" y="633.5" text-anchor="middle" fill="#b04a4a" font-weight="700">FAIL</text>
    <text x="40" y="658" fill="#12151a" font-weight="600">P7 · External causal-edge</text>
    <text x="250" y="658" fill="#5a616b">causal 0.559 &gt; null 0.500 but = twin 0.569 (Δ−0.010) · above null, not causal</text>
    <rect x="852" y="646" width="108" height="19" rx="9.5" fill="#faedd6"/><text x="906" y="659.5" text-anchor="middle" fill="#b6791f" font-weight="700">IN-DIST</text>
    <line x1="40" y1="676" x2="960" y2="676" stroke="#e2e6ea"/>
    <text x="40" y="700" fill="#12151a" font-weight="700">C2 · Do-operator control</text>
    <text x="250" y="700" fill="#2e8b57" font-weight="600">+0.118 condition / +0.162 gene vs the non-causal twin — the one positive</text>
    <rect x="852" y="688" width="108" height="19" rx="9.5" fill="#dcefe4"/><text x="906" y="701.5" text-anchor="middle" fill="#2e8b57" font-weight="700">POSITIVE</text>
  </g>
  <text x="40" y="736" font-size="13.5" fill="#3b424c">Under honest measurement the recoverable signal is far narrower than the genome-scale volume suggests: six probes sit at the floor, the one accuracy</text>
  <text x="40" y="753" font-size="13.5" fill="#3b424c">positive (C2) is <tspan font-weight="700">in-distribution, not causal</tspan>. The C2 anchor proves the nulls mean “no signal,” not “no sensitivity.”</text>
</svg>'''


def main():
    frac = frac_of_ceiling_gene()
    for model, *_ in ROWS:
        assert model in frac, f"{model} missing from fraction_of_ceiling.csv"
    svg = HEADER + "\n" + panel_a(frac) + "\n" + FOOTER + "\n"
    out = os.path.join(ROOT, "figures", "predictability_scorecard.svg")
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", out)
    print("Panel-A bars (fraction of ceiling, gene) from results/fraction_of_ceiling.csv:")
    for model, name, *_ in ROWS:
        print(f"  {name:22s} {frac[model]:.4f}")


if __name__ == "__main__":
    main()
