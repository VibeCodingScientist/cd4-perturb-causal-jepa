#!/usr/bin/env python
"""Generate figures/external_validation.svg from results/fusion_gf2.csv — the G-F.2 story:
the retrained do-operator recovers held-out external edge direction ABOVE null, but with NO advantage over
its non-causal twin → in-distribution, not causal. Data-driven; stdlib-only; text wrapped to fit the canvas.

Run:  python scripts/make_external_validation_figure.py
"""
import csv, io, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1000, 470
LO, HI = 0.40, 0.60             # y-axis window (bars cluster near 0.5–0.57)
PX_TOP, PX_BOT = 150, 340
def y(v): return PX_BOT - (v - LO) / (HI - LO) * (PX_BOT - PX_TOP)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(text, max_chars):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def tlines(text, x, y0, size, fill, max_chars, lh=None, weight=None, anchor="start"):
    lh = lh or int(size * 1.35)
    wt = f' font-weight="{weight}"' if weight else ""
    an = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return "\n".join(
        f'<text x="{x}" y="{y0 + i*lh}" font-size="{size}" fill="{fill}"{wt}{an}>{esc(line)}</text>'
        for i, line in enumerate(wrap(text, max_chars))), len(wrap(text, max_chars))


def rows():
    with io.open(os.path.join(ROOT, "results", "fusion_gf2.csv"), encoding="utf-8") as f:
        return {r["subset"]: r for r in csv.DictReader(f)}


def bar(x, w, v, color, label, bold=False):
    yy = y(v); h = PX_BOT - yy
    wt = ' font-weight="700"' if bold else ""
    return (f'<rect x="{x}" y="{yy:.1f}" width="{w}" height="{h:.1f}" rx="2" fill="{color}"/>'
            f'<text x="{x+w/2:.0f}" y="{yy-8:.1f}" text-anchor="middle" font-size="13" fill="{color}"{wt}>{v:.3f}</text>'
            f'<text x="{x+w/2:.0f}" y="{PX_BOT+20}" text-anchor="middle" font-size="12.5" fill="#3b424c">{label}</text>')


def main():
    r = rows()
    c = r["combined"]
    causal, twin, null = float(c["causal_acc"]), float(c["twin_acc"]), float(c["null_mean"])
    diff = float(c["causal_minus_twin"])
    fr, we = r["freimer(DE)"], r["weinstock(direct)"]
    sub, _ = tlines("Do-operator retrained with 10 CD4 TF regulators held OUT; zero-shot sign-recovery of "
                    "6,167 held-out external edges (Weinstock LLCB + Freimer KO-DE).",
                    40, 66, 13.5, "#4a515b", 118, lh=19)
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
         f'aria-label="External causal-edge validation G-F.2: causal {causal:.3f} versus non-causal twin '
         f'{twin:.3f} versus null {null:.3f}; recovers above null but not above the twin, in-distribution not causal." '
         f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
         '<text x="40" y="42" font-size="20" font-weight="700" fill="#12151a">External causal-edge validation (G-F.2): correlational, not causal</text>',
         sub,
         # axis + bars
         f'<line x1="100" y1="{PX_BOT}" x2="470" y2="{PX_BOT}" stroke="#c9d0da"/>',
         f'<line x1="100" y1="{y(0.5):.1f}" x2="470" y2="{y(0.5):.1f}" stroke="#8a8f98" stroke-dasharray="4 3"/>',
         f'<text x="96" y="{y(0.5)+4:.1f}" text-anchor="end" font-size="11.5" fill="#8a8f98">0.50 null</text>',
         f'<text x="96" y="{y(0.60)+4:.1f}" text-anchor="end" font-size="11.5" fill="#8a8f98">0.60</text>',
         f'<text x="96" y="{PX_BOT+4}" text-anchor="end" font-size="11.5" fill="#8a8f98">0.40</text>',
         '<text x="255" y="128" text-anchor="middle" font-size="14" font-weight="700" fill="#12151a">Combined held-out edges — sign accuracy</text>',
         bar(150, 74, null, "#9aa1ab", "null"),
         bar(240, 74, causal, "#2e8b57", "causal", True),
         bar(330, 74, twin, "#7e94c4", "twin"),
         # verdict callout (right, within canvas)
         '<rect x="600" y="150" width="360" height="192" rx="8" fill="#faedd6" stroke="#e0c98f"/>',
         '<text x="620" y="180" font-size="15" font-weight="700" fill="#9a6a15">IN-DISTRIBUTION, not causal</text>',
         f'<text x="620" y="206" font-size="13" fill="#5a4a1f">causal {causal:.3f} &gt; null {null:.3f} (p = 5×10⁻⁵),</text>',
         f'<text x="620" y="225" font-size="13" fill="#5a4a1f">but twin {twin:.3f} ≥ causal.</text>',
         f'<text x="620" y="251" font-size="14" font-weight="700" fill="#9a6a15">causal − twin = {diff:+.3f}</text>',
         '<text x="620" y="270" font-size="12.5" fill="#5a4a1f">regulator cluster-bootstrap 95% CI</text>',
         '<text x="620" y="287" font-size="12.5" fill="#5a4a1f">[−0.013, −0.005] — excludes 0.</text>',
         '<text x="620" y="313" font-size="12.5" fill="#5a4a1f">The within-dataset C2 advantage does</text>',
         '<text x="620" y="330" font-size="12.5" fill="#5a4a1f">NOT transfer to external causal edges.</text>',
         # per-source strip
         f'<text x="40" y="392" font-size="12.5" fill="#5a616b"><tspan font-weight="700">Freimer (DE), {fr["edges"]} edges:</tspan> causal {float(fr["causal_acc"]):.3f} vs twin {float(fr["twin_acc"]):.3f} — real above-null signal, no causal edge.</text>',
         f'<text x="40" y="414" font-size="12.5" fill="#5a616b"><tspan font-weight="700">Weinstock (direct), {we["edges"]} edges:</tspan> causal {float(we["causal_acc"]):.3f} (p = {float(we["p_causal_vs_null"]):.2f}) — at chance, underpowered (8 regulators).</text>',
         '<text x="40" y="450" font-size="11.5" fill="#8a8f98">results/fusion_gf2.csv · scripts/fusion_gf2.py + verify_gf2.py · sign-accuracy axis 0.40–0.60</text>',
         '</svg>']
    io.open(os.path.join(ROOT, "figures", "external_validation.svg"), "w", encoding="utf-8").write("\n".join(p) + "\n")
    print("wrote figures/external_validation.svg | causal", causal, "twin", twin, "null", null, "diff", diff)


if __name__ == "__main__":
    main()
