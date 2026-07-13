#!/usr/bin/env python
"""Generate figures/second_dataset_port.svg from results/gpa2_scorecard.csv — the Schmidt 2022 port:
the identical audit machinery runs on a second primary-cell dataset and produces a coherent, null-
discriminating scorecard (machinery PORTS), with the load-bearing caveat that R1 is cross-WELL (technical),
not the cross-DONOR biological floor Marson measured. Stdlib-only; text wrapped to fit the canvas.

Run:  python scripts/make_port_figure.py
"""
import csv, io, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1000, 470
PX_TOP, PX_BOT = 140, 330
def y(v): return PX_BOT - v * (PX_BOT - PX_TOP)       # 0..1 axis


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= n:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def tlines(text, x, y0, size, fill, n, lh):
    return "\n".join(f'<text x="{x}" y="{y0+i*lh}" font-size="{size}" fill="{fill}">{esc(l)}</text>'
                     for i, l in enumerate(wrap(text, n)))


def load():
    with io.open(os.path.join(ROOT, "results", "gpa2_scorecard.csv"), encoding="utf-8") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}


def bar(x, w, v, color, bold=False):
    yy = y(v); h = PX_BOT - yy
    wt = ' font-weight="700"' if bold else ""
    return (f'<rect x="{x}" y="{yy:.1f}" width="{w}" height="{h:.1f}" rx="2" fill="{color}"/>'
            f'<text x="{x+w/2:.0f}" y="{yy-6:.1f}" text-anchor="middle" font-size="12" fill="{color}"{wt}>{v:.2f}</text>')


PROBES = [("repro_floor_cross_well", ["R1 · reproducibility", "(cross-WELL)"]),
          ("reliability_ceiling_SB", ["R2 · reliability", "ceiling"]),
          ("relational_S", ["R3 · relational", "object S"])]


def main():
    d = load()
    ns, st = d["nostim"], d["stim"]
    sub = tlines("Identical probes + nulls, no retrain, on Schmidt's OWN recomputed floor (primary human T, "
                 "CRISPRa, 73 genes). All clear their permutation floor (p ≤ 1/501).",
                 40, 66, 13.5, "#4a515b", 116, 19)
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
         'aria-label="Second-dataset port to Schmidt 2022: three model-free probes reproduce above their null '
         'on Schmidt\'s own floor; the machinery ports, but R1 is cross-well technical not the cross-donor floor." '
         'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
         '<text x="40" y="42" font-size="20" font-weight="700" fill="#12151a">Second-dataset port (Schmidt 2022) — the audit machinery ports</text>',
         sub,
         # y axis
         f'<line x1="100" y1="{PX_TOP}" x2="100" y2="{PX_BOT}" stroke="#d5dae1"/>',
         f'<line x1="100" y1="{PX_BOT}" x2="470" y2="{PX_BOT}" stroke="#c9d0da"/>',
         f'<text x="94" y="{y(1.0)+4:.1f}" text-anchor="end" font-size="11.5" fill="#8a8f98">1.0</text>',
         f'<text x="94" y="{y(0.5)+4:.1f}" text-anchor="end" font-size="11.5" fill="#8a8f98">0.5</text>',
         f'<text x="94" y="{PX_BOT+4}" text-anchor="end" font-size="11.5" fill="#8a8f98">0</text>',
         # legend
         '<rect x="300" y="104" width="12" height="12" rx="2" fill="#3f9e6b"/><text x="318" y="114" font-size="12" fill="#5a616b">nostim</text>',
         '<rect x="384" y="104" width="12" height="12" rx="2" fill="#2e8b57"/><text x="402" y="114" font-size="12" fill="#5a616b">stim</text>']
    x0 = 130
    for i, (key, lbl) in enumerate(PROBES):
        gx = x0 + i * 116
        p.append(bar(gx, 42, float(ns[key]), "#3f9e6b"))
        p.append(bar(gx + 46, 42, float(st[key]), "#2e8b57", True))
        for j, line in enumerate(lbl):
            p.append(f'<text x="{gx+44}" y="{PX_BOT+18+j*15}" text-anchor="middle" font-size="12" fill="#3b424c">{esc(line)}</text>')
    # caveat callout (right, within canvas)
    p += ['<rect x="600" y="140" width="360" height="200" rx="8" fill="#f4dddb" stroke="#d9a9a4"/>',
          '<text x="620" y="168" font-size="15" font-weight="700" fill="#a8423e">Machinery ports — floor NOT re-tested</text>',
          '<text x="620" y="194" font-size="12.5" fill="#5a2a26"><tspan font-weight="700">R1 is cross-WELL</tspan> (technical replicate), not the</text>',
          '<text x="620" y="211" font-size="12.5" fill="#5a2a26">cross-DONOR biological floor Marson\'s 0.03</text>',
          '<text x="620" y="228" font-size="12.5" fill="#5a2a26">measured — the public form has no donor</text>',
          '<text x="620" y="245" font-size="12.5" fill="#5a2a26">demux. 0.71 vs 0.03 is a <tspan font-weight="700">non-comparison</tspan>.</text>',
          '<text x="620" y="271" font-size="12" fill="#5a2a26">Also: same lab · CRISPRa vs CRISPRi ·</text>',
          '<text x="620" y="288" font-size="12" fill="#5a2a26">selected strong hits · 3 of 7 probes</text>',
          '<text x="620" y="305" font-size="12" fill="#5a2a26">(P4/P5 N/A; P1/P2/P7 deferred).</text>',
          tlines("The instrument RUNS on a second, same-consortium primary-cell Perturb-seq dataset and produces "
                 "a coherent, null-discriminating scorecard — machinery portability, not evidence the "
                 "narrow-recoverable-signal finding generalizes. Next step: a donor-demuxed second dataset.",
                 40, 388, 12.5, "#5a616b", 116, 18),
          '<text x="40" y="450" font-size="11.5" fill="#8a8f98">results/gpa2_scorecard.csv · scripts/gpa2_stage2b_probes.py · GPA2_PORT.md</text>',
          '</svg>']
    io.open(os.path.join(ROOT, "figures", "second_dataset_port.svg"), "w", encoding="utf-8").write("\n".join(p) + "\n")
    print("wrote figures/second_dataset_port.svg")


if __name__ == "__main__":
    main()
