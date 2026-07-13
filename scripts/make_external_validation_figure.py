#!/usr/bin/env python
"""Generate figures/external_validation.svg from results/fusion_gf2.csv — the G-F.2 story:
the retrained do-operator recovers held-out external edge direction ABOVE null, but with NO advantage over
its non-causal twin → in-distribution, not causal. Data-driven; stdlib-only.

Run:  python scripts/make_external_validation_figure.py
"""
import csv, io, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LO, HI = 0.40, 0.60           # y-axis window (bars cluster near 0.5–0.57; window chosen to show it honestly)
PX_TOP, PX_BOT = 150, 340     # plot area (px)
def y(v): return PX_BOT - (v - LO) / (HI - LO) * (PX_BOT - PX_TOP)


def rows():
    with io.open(os.path.join(ROOT, "results", "fusion_gf2.csv"), encoding="utf-8") as f:
        return {r["subset"]: r for r in csv.DictReader(f)}


def bar(x, w, v, color, label, bold=False):
    yy = y(v); h = PX_BOT - yy
    wt = ' font-weight="700"' if bold else ""
    return (f'<rect x="{x}" y="{yy:.1f}" width="{w}" height="{h:.1f}" rx="2" fill="{color}"/>'
            f'<text x="{x+w/2:.0f}" y="{yy-7:.1f}" text-anchor="middle" font-size="13" fill="{color}"{wt}>{v:.3f}</text>'
            f'<text x="{x+w/2:.0f}" y="{PX_BOT+18}" text-anchor="middle" font-size="12.5" fill="#3b424c">{label}</text>')


def main():
    r = rows()
    c = r["combined"]
    causal, twin, null = float(c["causal_acc"]), float(c["twin_acc"]), float(c["null_mean"])
    diff = float(c["causal_minus_twin"])
    fr, we = r["freimer(DE)"], r["weinstock(direct)"]
    parts = [f'<svg viewBox="0 0 820 470" xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="External causal-edge validation G-F.2: do-operator causal {causal:.3f} vs non-causal twin {twin:.3f} vs null {null:.3f}; causal recovers above null but not above the twin, so the edge is in-distribution not causal." '
             f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">',
             '<rect x="0" y="0" width="820" height="470" fill="#ffffff"/>',
             '<text x="40" y="44" font-size="22" font-weight="700" fill="#12151a">External causal-edge validation (G-F.2) — recovery is correlational, not causal</text>',
             '<text x="40" y="70" font-size="13.5" fill="#4a515b">Do-operator retrained with 10 CD4 TF regulators held OUT; zero-shot sign-recovery of 6,167 held-out external edges (Weinstock LLCB + Freimer KO-DE).</text>',
             # axis
             f'<line x1="90" y1="{PX_BOT}" x2="470" y2="{PX_BOT}" stroke="#c9d0da"/>',
             f'<line x1="90" y1="{y(0.5):.1f}" x2="470" y2="{y(0.5):.1f}" stroke="#8a8f98" stroke-dasharray="4 3"/>',
             f'<text x="86" y="{y(0.5)+4:.1f}" text-anchor="end" font-size="11.5" fill="#8a8f98">0.50 null</text>',
             f'<text x="86" y="{y(0.60)+4:.1f}" text-anchor="end" font-size="11.5" fill="#8a8f98">0.60</text>',
             f'<text x="86" y="{PX_BOT+4}" text-anchor="end" font-size="11.5" fill="#8a8f98">0.40</text>',
             '<text x="255" y="118" text-anchor="middle" font-size="14" font-weight="700" fill="#12151a">Combined held-out edges — sign accuracy</text>',
             bar(140, 74, null, "#9aa1ab", "null"),
             bar(230, 74, causal, "#2e8b57", "causal", True),
             bar(320, 74, twin, "#7e94c4", "twin"),
             ]
    # verdict callout (right)
    parts += [
        '<rect x="500" y="150" width="290" height="190" rx="8" fill="#faedd6" stroke="#e0c98f"/>',
        '<text x="516" y="178" font-size="14" font-weight="700" fill="#9a6a15">IN-DISTRIBUTION, not causal</text>',
        f'<text x="516" y="204" font-size="12.5" fill="#5a4a1f">causal {causal:.3f} &gt; null {null:.3f} (p=5×10⁻⁵)</text>',
        f'<text x="516" y="224" font-size="12.5" fill="#5a4a1f">but twin {twin:.3f} ≥ causal</text>',
        f'<text x="516" y="248" font-size="13.5" font-weight="700" fill="#9a6a15">causal − twin = {diff:+.3f}</text>',
        '<text x="516" y="268" font-size="12" fill="#5a4a1f">regulator cluster-bootstrap</text>',
        '<text x="516" y="285" font-size="12" fill="#5a4a1f">95% CI [−0.013, −0.005] (excludes 0)</text>',
        '<text x="516" y="312" font-size="12" fill="#5a4a1f">The within-dataset C2 advantage does</text>',
        '<text x="516" y="328" font-size="12" fill="#5a4a1f">NOT transfer to external causal edges.</text>',
    ]
    # per-source strip
    parts += [
        f'<text x="40" y="392" font-size="12.5" fill="#5a616b"><tspan font-weight="700">Freimer (DE), {fr["edges"]} edges:</tspan> causal {float(fr["causal_acc"]):.3f} vs twin {float(fr["twin_acc"]):.3f} — real above-null signal, no causal edge.</text>',
        f'<text x="40" y="414" font-size="12.5" fill="#5a616b"><tspan font-weight="700">Weinstock (direct), {we["edges"]} edges:</tspan> causal {float(we["causal_acc"]):.3f} (p={float(we["p_causal_vs_null"]):.2f}) — at chance, underpowered (8 regulators); not over-read.</text>',
        '<text x="40" y="450" font-size="11.5" fill="#8a8f98">results/fusion_gf2.csv · scripts/fusion_gf2.py + verify_gf2.py · axis window 0.40–0.60</text>',
        '</svg>',
    ]
    out = os.path.join(ROOT, "figures", "external_validation.svg")
    io.open(out, "w", encoding="utf-8").write("\n".join(parts) + "\n")
    print("wrote", out, "| causal", causal, "twin", twin, "null", null, "diff", diff)


if __name__ == "__main__":
    main()
