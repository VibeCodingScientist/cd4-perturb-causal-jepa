#!/usr/bin/env python
"""Generate figures/second_dataset_port.svg from results/gpa2_scorecard.csv — the Schmidt 2022 port:
the identical audit machinery runs on a second primary-cell dataset and produces a coherent, null-
discriminating scorecard (machinery PORTS). With the load-bearing caveat that R1 is cross-WELL (technical),
not the cross-DONOR biological floor Marson measured — so the floor finding is NOT re-tested. Stdlib-only.

Run:  python scripts/make_port_figure.py
"""
import csv, io, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PX_TOP, PX_BOT = 130, 330
def y(v): return PX_BOT - v * (PX_BOT - PX_TOP)      # 0..1 axis


def load():
    with io.open(os.path.join(ROOT, "results", "gpa2_scorecard.csv"), encoding="utf-8") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}


def bar(x, w, v, color, bold=False):
    yy = y(v); h = PX_BOT - yy
    wt = ' font-weight="700"' if bold else ""
    return (f'<rect x="{x}" y="{yy:.1f}" width="{w}" height="{h:.1f}" rx="2" fill="{color}"/>'
            f'<text x="{x+w/2:.0f}" y="{yy-6:.1f}" text-anchor="middle" font-size="12" fill="{color}"{wt}>{v:.2f}</text>')


PROBES = [("repro_floor_cross_well", "R1 · reproducibility\n(cross-WELL)"),
          ("reliability_ceiling_SB", "R2 · reliability\nceiling"),
          ("relational_S", "R3 · relational\nobject S")]


def main():
    d = load()
    ns, st = d["nostim"], d["stim"]
    parts = ['<svg viewBox="0 0 820 470" xmlns="http://www.w3.org/2000/svg" role="img" '
             'aria-label="Second-dataset port to Schmidt 2022: three model-free probes reproduce above their null on Schmidt\'s own floor; the machinery ports, but R1 is cross-well technical replicate not the cross-donor biological floor, so the floor finding is not re-tested." '
             'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">',
             '<rect x="0" y="0" width="820" height="470" fill="#ffffff"/>',
             '<text x="40" y="44" font-size="22" font-weight="700" fill="#12151a">Second-dataset port (Schmidt 2022) — the audit machinery ports (qualified)</text>',
             '<text x="40" y="70" font-size="13.5" fill="#4a515b">Identical probes + nulls, no retrain, on Schmidt\'s OWN recomputed floor (primary human T, CRISPRa, 73 genes). All clear their permutation floor (p ≤ 1/501).</text>',
             # y axis
             f'<line x1="90" y1="{PX_TOP}" x2="90" y2="{PX_BOT}" stroke="#d5dae1"/>',
             f'<line x1="90" y1="{PX_BOT}" x2="470" y2="{PX_BOT}" stroke="#c9d0da"/>',
             f'<text x="84" y="{y(1.0)+4:.1f}" text-anchor="end" font-size="11.5" fill="#8a8f98">1.0</text>',
             f'<text x="84" y="{y(0.5)+4:.1f}" text-anchor="end" font-size="11.5" fill="#8a8f98">0.5</text>',
             f'<text x="84" y="{PX_BOT+4}" text-anchor="end" font-size="11.5" fill="#8a8f98">0</text>',
             # legend
             '<rect x="300" y="96" width="12" height="12" rx="2" fill="#3f9e6b"/><text x="318" y="106" font-size="12" fill="#5a616b">nostim</text>',
             '<rect x="380" y="96" width="12" height="12" rx="2" fill="#2e8b57"/><text x="398" y="106" font-size="12" fill="#5a616b">stim</text>',
             ]
    x0 = 120
    for i, (key, lbl) in enumerate(PROBES):
        gx = x0 + i * 118
        parts.append(bar(gx, 42, float(ns[key]), "#3f9e6b"))
        parts.append(bar(gx + 46, 42, float(st[key]), "#2e8b57", True))
        for j, line in enumerate(lbl.split("\n")):
            parts.append(f'<text x="{gx+44}" y="{PX_BOT+18+j*15}" text-anchor="middle" font-size="12" fill="#3b424c">{line}</text>')
    # caveat callout (right)
    parts += [
        '<rect x="500" y="130" width="290" height="200" rx="8" fill="#f4dddb" stroke="#d9a9a4"/>',
        '<text x="516" y="158" font-size="14" font-weight="700" fill="#a8423e">Machinery ports — floor NOT re-tested</text>',
        '<text x="516" y="184" font-size="12.5" fill="#5a2a26"><tspan font-weight="700">R1 is cross-WELL</tspan> (technical replicate),</text>',
        '<text x="516" y="201" font-size="12.5" fill="#5a2a26">not the cross-DONOR biological floor</text>',
        '<text x="516" y="218" font-size="12.5" fill="#5a2a26">Marson\'s 0.03 measured — the public form</text>',
        '<text x="516" y="235" font-size="12.5" fill="#5a2a26">has no donor demux. 0.71 vs 0.03 is a</text>',
        '<text x="516" y="252" font-size="12.5" fill="#5a2a26"><tspan font-weight="700">non-comparison</tspan>.</text>',
        '<text x="516" y="278" font-size="12" fill="#5a2a26">Also: same lab · CRISPRa vs CRISPRi ·</text>',
        '<text x="516" y="294" font-size="12" fill="#5a2a26">selected strong hits · 3/7 probes</text>',
        '<text x="516" y="311" font-size="12" fill="#5a2a26">(P4/P5 N/A; P1/P2/P7 deferred).</text>',
    ]
    parts += [
        '<text x="40" y="392" font-size="12.5" fill="#5a616b">The instrument RUNS on a second, same-consortium primary-cell Perturb-seq dataset and produces a coherent, null-discriminating scorecard.</text>',
        '<text x="40" y="412" font-size="12.5" fill="#5a616b">That is <tspan font-weight="700">machinery portability</tspan> — not evidence the narrow-recoverable-signal finding generalizes. Next step: a donor-demuxed second dataset.</text>',
        '<text x="40" y="450" font-size="11.5" fill="#8a8f98">results/gpa2_scorecard.csv · scripts/gpa2_stage2b_probes.py · GPA2_PORT.md</text>',
        '</svg>',
    ]
    out = os.path.join(ROOT, "figures", "second_dataset_port.svg")
    io.open(out, "w", encoding="utf-8").write("\n".join(parts) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
