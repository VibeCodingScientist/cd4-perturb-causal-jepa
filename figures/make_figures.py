"""figures.make_figures — the three demo figures + biology annotation (§12).

  Figure 1  benchmark table: Pearson-delta / PerturBench-rank (red = mode-collapse) /
            DES, with the best row per hold-out highlighted.
  Figure 2  the 2x2 on condition-hold-out Pearson-delta — reads out the do-operator
            (C2) and JEPA-init (C3) effects in one chart.
  Figure 3  sample-efficiency curve: random vs VOI-guided subsampling, with the
            90%-of-full-screen point annotated.
  Figure 4  (if-time) biology: top-20 VOI-disagreement genes x family x T-cell role x
            GWAS flag.

Every function takes already-computed data (a DataFrame / grid / SubsamplingCurve) and
an output path, so tests render from mock artifacts with no trained models. The
``make_all_figures`` orchestrator wires them to the committed artifacts + the ablation
/ VOI layers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import contract  # noqa: E402

# theme-neutral palette
_C_RANDOM = "#9aa0a6"
_C_JEPA = "#4c78a8"
_C_MASK_ON = "#e45756"
_C_OK = "#54a24b"
_C_RED = "#d62728"


def _save(fig, out_path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Figure 1 — benchmark table
# ---------------------------------------------------------------------------
def figure1_benchmark(
    benchmark: pd.DataFrame,
    out_path=None,
    metrics: Optional[list] = None,
) -> str:
    """Render the headline benchmark table. Mode-collapsed rows (perturbench_rank >
    threshold) are drawn red; the best Pearson-delta per split is highlighted green."""
    out_path = out_path or (contract.FIGURES_DIR / "figure1_benchmark.png")
    metrics = metrics or list(contract.METRICS_HEADLINE)
    df = benchmark.copy()
    show_cols = ["model", "split"] + metrics + [contract.MODE_COLLAPSE_FLAG]
    show_cols = [c for c in show_cols if c in df.columns]
    df = df[show_cols].sort_values(["split", "model"]).reset_index(drop=True)

    # best (highest) Pearson-delta per split -> highlight
    best_idx = set()
    if contract.METRIC_PEARSON_DELTA in df.columns:
        for _split, g in df.groupby("split"):
            best_idx.add(g[contract.METRIC_PEARSON_DELTA].idxmax())

    fig, ax = plt.subplots(figsize=(1.6 * len(show_cols), 0.5 * len(df) + 1.2))
    ax.axis("off")
    ax.set_title("Figure 1 — Benchmark (headline metrics)", fontsize=12, loc="left", pad=12)

    cell_text, cell_colors = [], []
    for i, row in df.iterrows():
        collapsed = bool(row.get(contract.MODE_COLLAPSE_FLAG, False))
        texts, colors = [], []
        for c in show_cols:
            val = row[c]
            if isinstance(val, float):
                texts.append(f"{val:.3f}" if pd.notna(val) else "—")
            else:
                texts.append(str(val))
            if collapsed:
                colors.append("#f8d7da")
            elif i in best_idx:
                colors.append("#d7f0d0")
            else:
                colors.append("white")
        cell_text.append(texts)
        cell_colors.append(colors)

    table = ax.table(
        cellText=cell_text, colLabels=show_cols, cellColours=cell_colors,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for j, c in enumerate(show_cols):  # header styling
        table[0, j].set_text_props(weight="bold")
    # note
    fig.text(
        0.01, 0.01,
        "red = mode-collapsed (perturbench_rank > %.1f); green = best Pearson-δ per hold-out"
        % contract.MODE_COLLAPSE_THRESHOLD,
        fontsize=8, color="#555",
    )
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 2 — the 2x2
# ---------------------------------------------------------------------------
def figure2_2x2(grid: pd.DataFrame, out_path=None, c2=None, c3=None) -> str:
    """Grouped bars of the 2x2 (condition hold-out Pearson-delta). Groups = encoder
    init (random / JEPA); bars within a group = causal mask off / on. C2 and C3 are
    annotated if provided (from ``core.ablation``)."""
    out_path = out_path or (contract.FIGURES_DIR / "figure2_2x2.png")
    inits = list(grid.index)          # ["random", "jepa"]
    masks = list(grid.columns)        # ["off", "on"]
    x = np.arange(len(inits))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, mask in enumerate(masks):
        vals = [grid.loc[init, mask] for init in inits]
        color = _C_MASK_ON if mask == "on" else _C_RANDOM
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=f"mask {mask}", color=color)
        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}-init" for i in inits])
    ax.set_ylabel("Condition hold-out Pearson-δ (top-50 DEGs)")
    ax.set_title("Figure 2 — 2×2 ablation: do-operator (C2) × JEPA-init (C3)")
    ax.legend(title="Causal mask", loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    caption = []
    if c2 is not None:
        caption.append(str(c2))
    if c3 is not None:
        caption.append(str(c3))
    if caption:
        fig.text(0.01, -0.02, "   ".join(caption), fontsize=9, color="#333")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 3 — sample-efficiency curve
# ---------------------------------------------------------------------------
def figure3_subsampling(curve, out_path=None) -> str:
    """Random (mean ± std) vs VOI-guided subsampling; 90%-of-full point annotated."""
    out_path = out_path or (contract.FIGURES_DIR / "figure3_subsampling.png")
    fr = np.array(curve.fractions) * 100.0
    rmean = np.array(curve.random_mean)
    rstd = np.array(curve.random_std)
    voi = np.array(curve.voi_score)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(fr, rmean, "-o", color=_C_RANDOM, label="random selection")
    ax.fill_between(fr, rmean - rstd, rmean + rstd, color=_C_RANDOM, alpha=0.2)
    if np.isfinite(voi).any():
        ax.plot(fr, voi, "-s", color=_C_JEPA, label="VOI-guided selection")

    target = 0.90 * curve.full_score
    ax.axhline(target, ls="--", color=_C_OK, alpha=0.8, label="90% of full-screen")
    for frac_attr, color, name in [
        (getattr(curve, "voi_90_fraction", None), _C_JEPA, "VOI"),
        (getattr(curve, "random_90_fraction", None), _C_RANDOM, "random"),
    ]:
        if frac_attr is not None:
            ax.axvline(frac_attr * 100.0, ls=":", color=color, alpha=0.7)
            ax.annotate(
                f"{name}: {frac_attr*100:.0f}%",
                xy=(frac_attr * 100.0, target),
                xytext=(frac_attr * 100.0 + 3, target - 0.05 * max(1e-6, curve.full_score)),
                fontsize=9, color=color,
            )

    ax.set_xlabel("% of training perturbations measured")
    ax.set_ylabel("Condition hold-out Pearson-δ")
    ax.set_title("Figure 3 — Sample efficiency: VOI-guided vs random")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 4 — biology annotation (if-time)
# ---------------------------------------------------------------------------
def build_biology_annotation(
    gene_disagreement: pd.Series,
    annotations: Optional[pd.DataFrame] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Join the top-N VOI-disagreement genes with family / T-cell role / GWAS flag.

    ``annotations`` is a DataFrame indexed by gene with columns
    ['gene_family', 'tcell_role', 'gwas_flag']; missing entries become 'unknown' /
    False. The real annotation source is a reference table joined in Lane C; the
    figure just needs this shape.
    """
    top = gene_disagreement.sort_values(ascending=False).head(top_n)
    out = pd.DataFrame({"gene": top.index, "disagreement": top.values})
    if annotations is not None:
        ann = annotations.reindex(top.index)
        out["gene_family"] = ann.get("gene_family", pd.Series(index=top.index)).fillna("unknown").values
        out["tcell_role"] = ann.get("tcell_role", pd.Series(index=top.index)).fillna("unknown").values
        out["gwas_flag"] = ann.get("gwas_flag", pd.Series(index=top.index)).fillna(False).astype(bool).values
    else:
        out["gene_family"] = "unknown"
        out["tcell_role"] = "unknown"
        out["gwas_flag"] = False
    return out.reset_index(drop=True)


def figure4_biology(annotation: pd.DataFrame, out_path=None) -> str:
    """Horizontal bar of top-N genes by VOI disagreement, colored by GWAS flag, with
    family / T-cell role annotated."""
    out_path = out_path or (contract.FIGURES_DIR / "figure4_biology.png")
    df = annotation.iloc[::-1].reset_index(drop=True)  # highest at top
    colors = [_C_RED if g else _C_JEPA for g in df["gwas_flag"]]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(df) + 1.5))
    ax.barh(np.arange(len(df)), df["disagreement"], color=colors)
    ax.set_yticks(np.arange(len(df)))
    labels = [
        f"{g}  ({fam} · {role})" for g, fam, role in
        zip(df["gene"], df["gene_family"], df["tcell_role"])
    ]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Ensemble VOI disagreement")
    ax.set_title("Figure 4 — Top VOI-disagreement genes (red = GWAS-flagged)")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def make_all_figures(
    benchmark: Optional[pd.DataFrame] = None,
    grid: Optional[pd.DataFrame] = None,
    curve=None,
    biology: Optional[pd.DataFrame] = None,
    c2=None,
    c3=None,
) -> dict:
    """Render whatever inputs are available; returns {figure_name: path}. Reads
    ``results/benchmark_table.csv`` for Figure 1 if ``benchmark`` is not supplied."""
    paths = {}
    if benchmark is None and contract.BENCHMARK_TABLE.exists():
        benchmark = pd.read_csv(contract.BENCHMARK_TABLE)
    if benchmark is not None and len(benchmark):
        paths["figure1"] = figure1_benchmark(benchmark)
    if grid is not None:
        paths["figure2"] = figure2_2x2(grid, c2=c2, c3=c3)
    if curve is not None:
        paths["figure3"] = figure3_subsampling(curve)
    if biology is not None:
        paths["figure4"] = figure4_biology(biology)
    return paths


__all__ = [
    "figure1_benchmark", "figure2_2x2", "figure3_subsampling",
    "build_biology_annotation", "figure4_biology", "make_all_figures",
]
